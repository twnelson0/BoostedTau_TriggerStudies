import awkward as ak
import uproot
import hist
from hist import intervals
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
from coffea import processor, nanoevents
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import candidate
from math import pi	

def deltaR(part1, part2):
	return np.sqrt((part2.eta - part1.eta)**2 + (part2.phi - part1.phi)**2)

def mass(part1,part2):
	return np.sqrt((part1.E + part2.E)**2 - (part1.Px + part2.Px)**2 - (part1.Py + part2.Py)**2 - (part1.Pz + part2.Pz)**2)

#def mass_pt(part1,part2):
#	return np.sqrt(part1.mt*np.cosh(part1.eta) + )**2 - 

def bit_mask(in_bits):
	mask = 0
	for bit in in_bits:
		mask += (1 << bit)
	return mask


class TriggerStudies(processor.ProcessorABC):
	def __init__(self, trigger_bit, signal = True):
		self.trigger_bit = trigger_bit
		self.signal = signal
		#pass
	
	def process(self, events):
		dataset = events.metadata['dataset']
		tau = ak.zip( 
			{
				"pt": events.boostedTauPt,
				"E": events.boostedTauEnergy,
				"Px": events.boostedTauPx,
				"Py": events.boostedTauPy,
				"Pz": events.boostedTauPz,
				"mass": events.boostedTauMass,
				"eta": events.boostedTauEta,
				"phi": events.boostedTauPhi,
				"nBoostedTau": events.nBoostedTau,
				"charge": events.boostedTauCharge,
				"iso1": events.boostedTauByIsolationMVArun2v1DBoldDMwLTrawNew,
				"iso2": events.boostedTaupfTausDiscriminationByDecayModeFinding,
				"trigger": events.HLTJet,
				"pfMET": events.pfMET,
			},
			with_name="TauArray",
			behavior=candidate.behavior,
		)

		AK8Jet = ak.zip(
			{
				"AK8JetDropMass": events.AK8JetSoftDropMass,
				"AK8JetPt": events.AK8JetPt,
				"nMu": events.nMu,
				"nEle": events.nEle,
				"trigger": events.HLTJet,
			},
			with_name="AK8JetArray",
			behavior=candidate.behavior,
		)
		
		Muon = ak.zip(
			{
				"nMu": events.nMu,
				"E": events.muEn,
				"px": events.muPt*np.cos(events.muPhi),
				"py": events.muPt*np.sin(events.muPhi),
				"px": events.muPt*np.sinh(events.muEta),
				"charge": events.muCharge,
				"trigger": events.HLTJet,
			},
			with_name="AK8JetArray",
			behavior=candidate.behavior,
		)
		
		Electron = ak.zip(
			{
				"nEle": events.nEle,
				"E": events.eleEn,
				"px": events.elePt*np.cos(events.elePhi),
				"py": events.elePt*np.sin(events.elePhi),
				"px": events.elePt*np.sinh(events.eleEta),
				"charge": events.eleCharge,
				"trigger": events.HLTJet,
			},
			with_name="AK8JetArray",
			behavior=candidate.behavior,
		)
		
		Jet = ak.zip(
			{
				"JetPt": events.jetPt,
				"pfMET": events.pfMET,
				"HT": ak.sum(events.jetPt, axis=1),
				"eta": events.jetEta,
				"trigger": events.HLTJet,
			},
			with_name="PFJetArray",
			behavior=candidate.behavior,
		)
		
		#Histograms (AK8Jet) (Trigger bit = 40)
		AK8Pt_all = hist.Hist.new.StrCat(["No Trigger","Trigger"], name = "AK8Pt_hist").Reg(40,0,1100, name="AK8Pt", label = "AK8 Jet r$p_T$ [GeV]").Double()	
		AK8SoftMass_all = hist.Hist.new.StrCat(["No Trigger","Trigger"], name = "AK8SoftMass_hist").Reg(40,0,400, name="AK8SoftMass", label = "AK8 Jet Soft Drop Mass [GeV]").Double()
		AK8Pt_PreTrigg = hist.Hist.new.Reg(40, 0, 1100, name = "JetPt_Trigg", label = r"AK8Jet $p_T$ [GeV]").Double()
		AK8Pt_Trigg = hist.Hist.new.Reg(40, 0, 1100, name = "JetPt_Trigg", label = r"AK8Jet $p_T$ [GeV]").Double()
		AK8SoftMass_PreTrigg = hist.Hist.new.Reg(40, 0, 300, name = "SoftMass_Trigg", label = "AK8Jet Soft Mass [GeV]").Double()
		AK8SoftMass_Trigg = hist.Hist.new.Reg(40, 0, 300, name = "SoftMass_Trigg", label = "AK8Jet Soft Mass [GeV]").Double()		
		
		#2D Histograms
		AK8Jet_PreTrigger = hist.Hist(
						hist.axis.Regular(20, 0, 1100, name="JetPt", label=r"AK8Jet $p_T$ [GeV]", flow=False),
						hist.axis.Regular(10, 0, 300, name="SoftMass", label=r"AK8Jet Soft Mass [GeV]", flow=False)
					)		
		AK8Jet_Trigger = hist.Hist(
						hist.axis.Regular(20, 0, 1100, name="JetPt", label=r"AK8Jet $p_T$ [GeV]", flow=False),
						hist.axis.Regular(10, 0, 300, name="SoftMass", label="AK8Jet Soft Mass [GeV]", flow=False)
					)		
		eff_AK8Jet = hist.Hist(
						hist.axis.Regular(20, 0, 1100, name="JetPt", label=r"AK8Jet $p_T$ [GeV]", flow=False),
						hist.axis.Regular(10, 0, 300, name="SoftMass", label="AK8Jet Soft Mass [GeV]", flow=False)
					)		

		#Histograms (MET and HT) (Trigger bit = 39)
		HT_PreTrigg = hist.Hist.new.Reg(40, 0, 2000., label = "HT [GeV]").Double()
		HT_Trigg = hist.Hist.new.Reg(40, 0, 2000., label = "HT [GeV]").Double()
		MET_PreTrigg = hist.Hist.new.Reg(30, 0, 1000., name="MET", label="MET [GeV]").Double()
		MET_Trigg = hist.Hist.new.Reg(30, 0, 1000., name="MET", label="MET [GeV]").Double()

		#2D Histograms
		Jet_PreTrigger = hist.Hist(
			hist.axis.Regular(20, 0, 1000., name = "pfMET" , label = r"MET [GeV]"),
			hist.axis.Regular(20, 0, 2000., name = "HT", label = r"HT [GeV]")
		)
		Jet_Trigger = hist.Hist(
			hist.axis.Regular(20, 0, 1000., name = "pfMET" , label = r"MET [GeV]"),
			hist.axis.Regular(20, 0, 2000., name = "HT", label = r"HT [GeV]")
		)
		eff_Jet = hist.Hist(
			hist.axis.Regular(20, 0, 1000., name = "pfMET" , label = r"MET [GeV]"),
			hist.axis.Regular(20, 0, 2000., name = "HT", label = r"HT [GeV]")
		)
	

		trigger_mask = bit_mask([self.trigger_bit])		
		tau = tau[tau.pt > 30] #pT
		tau = tau[tau.eta < 2.3] #eta
		
		#Loose isolation
		tau = tau[tau.iso1 >= 0.5]
		tau = tau[tau.iso2 >= 0.5]		
		
		AK8Jet = AK8Jet[(ak.sum(tau.charge,axis=1) == 0)] #Apply charge conservation cut to AK8Jets
		tau = tau[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		Muon = Muon[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		Electron = Electron[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation

		AK8Jet = AK8Jet[ak.num(tau) == 4]
		Electron = Electron[ak.num(tau) == 4]
		Muon = Muon[ak.num(tau) == 4]
		tau = tau[ak.num(tau) == 4] #4 tau events

		#Obtain electron and muon pairings
		

		if (self.trigger_bit == 40):
			#Cut Z-->2mu and Z-->2e events
			if (not(self.signal)):
				print("%d Events"%len(ak.ravel(AK8Jet.AK8JetPt)))
				for x,y in zip(ak.ravel(Muon.nMu), ak.ravel(Electron.nEle)):
					if (x > 0):
						print("%d muons"%x)
					if (y > 0):
						print("%d electrons"%y)
					print("%d leptons"%(x + y))
				
					
				#AK8Jet = AK8Jet[AK8Jet.nMu == 0]	
				#print("%d Events"%len(ak.ravel(AK8Jet.nMu)))
				#AK8Jet = AK8Jet[AK8Jet.nEle == 0]	
				#print("%d Events"%len(ak.ravel(AK8Jet.nMu)))
			AK8Pt_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetPt))
			AK8Pt_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetPt)
			AK8SoftMass_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetDropMass))
			AK8SoftMass_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetDropMass)
			AK8Pt_all.fill("No Trigger",ak.ravel(AK8Jet.AK8JetPt))
		
		if (self.trigger_bit == 39):
			#Apply Jet Cuts
			Jet = Jet[Jet.eta <= 3]	
			Jet = Jet[Jet.HT > 30]
			
			#Fill Histograms
			HT_PreTrigg.fill(ak.ravel(Jet.HT))
			HT_NoTrigg_Arr = ak.ravel(Jet.HT)
			MET_PreTrigg.fill(ak.ravel(Jet.pfMET))
			MET_NoTrigg_Arr = ak.ravel(Jet.pfMET)	


		tau = tau[np.bitwise_and(tau.trigger,trigger_mask) == trigger_mask]
		AK8Jet = AK8Jet[np.bitwise_and(AK8Jet.trigger,trigger_mask) == trigger_mask]
		Jet = Jet[np.bitwise_and(Jet.trigger,trigger_mask) == trigger_mask]
		tau_plus = tau[tau.charge > 0]	
		tau_minus = tau[tau.charge < 0]

		#Construct all possible valid ditau pairs
		tau_plus1, tau_plus2 = ak.unzip(ak.combinations(tau_plus,2))
		tau_minus1, tau_minus2 = ak.unzip(ak.combinations(tau_minus,2))
		
		deltaR11 = deltaR(tau_plus1, tau_minus1)
		deltaR12 = deltaR(tau_plus1, tau_minus2)
		deltaR22 = deltaR(tau_plus2, tau_minus2)
		deltaR21 = deltaR(tau_plus2, tau_minus1)
		
		#Efficiency Histograms (How do I do these??)
		if (self.trigger_bit == 40):	
			AK8Pt_Trigg.fill(ak.ravel(AK8Jet.AK8JetPt))
			AK8SoftMass_Trigg.fill(ak.ravel(AK8Jet.AK8JetDropMass))
			AK8Pt_all.fill("Trigger",ak.ravel(AK8Jet.AK8JetPt))	
			AK8Pt_Trigg_Arr = ak.ravel(AK8Jet.AK8JetPt)
			AK8SoftMass_all.fill("Trigger",ak.ravel(AK8Jet.AK8JetDropMass))	
			AK8SoftMass_Trigg_Arr = ak.ravel(AK8Jet.AK8JetDropMass)
	
			print("Efficiency (AK8Jet Trigger): %f"%(ak.num(AK8Pt_Trigg_Arr,axis=0)/ak.num(AK8Pt_NoTrigg_Arr,axis=0)))
			AK8Jet_PreTrigger.fill(AK8Pt_NoTrigg_Arr, AK8SoftMass_NoTrigg_Arr)
			AK8Jet_Trigger.fill(AK8Pt_Trigg_Arr, AK8SoftMass_Trigg_Arr)
			eff_AK8Jet = AK8Jet_Trigger/AK8Jet_PreTrigger
		
		if (self.trigger_bit == 39):
			HT_Trigg.fill(ak.ravel(Jet.HT))
			HT_Trigg_Arr = ak.ravel(Jet.HT)
			MET_Trigg.fill(ak.ravel(Jet.pfMET))
			MET_Trigg_Arr = ak.ravel(Jet.pfMET)	
			
			print("Efficiency (HT+MET Trigger): %f"%(ak.num(MET_Trigg_Arr,axis=0)/ak.num(MET_NoTrigg_Arr,axis=0)))
			Jet_PreTrigger.fill(MET_NoTrigg_Arr, HT_NoTrigg_Arr)
			Jet_Trigger.fill(MET_Trigg_Arr, HT_Trigg_Arr)
			eff_Jet = Jet_Trigger/Jet_PreTrigger
		
		if (self.trigger_bit == 40):
			return{
				 dataset: {
					"AK8JetPt_PreTrigg": AK8Pt_PreTrigg,
					"AK8JetPt_Trigg": AK8Pt_Trigg,
					"AK8JetSoftMass_PreTrigg": AK8SoftMass_PreTrigg,
					"AK8JetSoftMass_Trigg": AK8SoftMass_Trigg,
					"AK8Jet_PreTrigg": AK8Jet_PreTrigger,
					"AK8Jet_Trigg": AK8Jet_Trigger,
					"AK8Jet_eff": eff_AK8Jet,
				}
			}
		if (self.trigger_bit == 39):
			return{
				 dataset: {
					"MET_PreTrigg": MET_PreTrigg,
					"MET_Trigg": MET_Trigg,
					"HT_PreTrigg": HT_PreTrigg,
					"HT_Trigg": HT_Trigg,
					"Jet_PreTrigg": Jet_PreTrigger,
					"Jet_Trigg": Jet_Trigger,
					"Jet_eff": eff_Jet,
				}
			}

	
	def postprocess(self, accumulator):
		pass	

class TauPlotting(processor.ProcessorABC):
	def __init__(self):
		pass
	
	def process(self, events):
		dataset = events.metadata['dataset']
		tau = ak.zip( 
			{
				"pt": events.boostedTauPt,
				"E": events.boostedTauEnergy,
				"Px": events.boostedTauPx,
				"Py": events.boostedTauPy,
				"Pz": events.boostedTauPz,
				"mass": events.boostedTauMass,
				"eta": events.boostedTauEta,
				"phi": events.boostedTauPhi,
				"leadingIndx": events.leadtauIndex,
				"nBoostedTau": events.nBoostedTau,
				"charge": events.boostedTauCharge,
				"iso1": events.boostedTauByIsolationMVArun2v1DBoldDMwLTrawNew,
				"iso2": events.boostedTaupfTausDiscriminationByDecayModeFinding,
			},
			with_name="TauArray",
			behavior=candidate.behavior,
		)
		
		pt_hist = (
			hist.Hist.new
            .Reg(50, 0, 1000., name="p_T", label="$p_{T}$ [GeV]") 
            .Int64()
		)
		E_hist = (
			hist.Hist.new
            .Reg(50, 0, 500., name="E", label=r"Energy [GeV]") 
            .Int64()
		)
		pt_all_hist = (
			hist.Hist.new
			.StrCat(["Leading","Subleading","Third-leading","Fourth-leading"], name = "tau_pt")
            .Reg(50, 0, 1500., name="p_T_all", label="$p_{T}$ [GeV]") 
            .Int64()
		)
		eta_hist = (
			hist.Hist.new
            .Reg(50, -3., 3., name="eta", label=r"$\eta$")
            .Int64()
		)
		phi_hist = (
			hist.Hist.new
            .Reg(50, -pi, pi, name="phi", label="$phi$")
            .Int64()
		)
		pt4_hist = (
			hist.Hist.new
            .Reg(50, 0, 500., name="p_T4", label="$p_{T}$ [GeV]") 
            .Int64()
		)
		ditau_mass1_hist = (
			hist.Hist.new
			.Reg(50, 0, 200., name = "mass1", label=r"$m_{\tau \tau} [GeV]$")
			.Double()
		)
		ditau_mass2_hist = (
			hist.Hist.new
			.Reg(50, 0, 200., name = "mass2", label=r"$m_{\tau \tau} [GeV]$")
			.Double()
		)
		dimass_all_hist = (
			hist.Hist.new
			.StrCat(["Leading pair","Subleading pair"], name = "ditau_mass")
			.Reg(50, 0, 200., name="ditau_mass_all", label=r"$m_{\tau\tau}$ [GeV]") 
            .Double()
		)
			
		#Apply cuts/selection
		tau = tau[tau.pt > 30] #pT
		tau = tau[tau.eta < 2.3] #eta
		
		#Loose isolation
		tau = tau[tau.iso1 >= 0.5]
		tau = tau[tau.iso2 >= 0.5]		
		
		tau = tau[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		
		#print("Before 4 tau cut length is: %d" % len(tau))
		tau = tau[ak.num(tau) == 4] #4 tau events (unsure about this)	
		#print("After 4 tau cut length is: %d" % len(tau))
		tau_plus = tau[tau.charge > 0]	
		tau_minus = tau[tau.charge < 0]

		#Construct all possible valid ditau pairs
		tau_plus1, tau_plus2 = ak.unzip(ak.combinations(tau_plus,2))
		tau_minus1, tau_minus2 = ak.unzip(ak.combinations(tau_minus,2))
		
		deltaR11 = deltaR(tau_plus1, tau_minus1)
		deltaR12 = deltaR(tau_plus1, tau_minus2)
		deltaR22 = deltaR(tau_plus2, tau_minus2)
		deltaR21 = deltaR(tau_plus2, tau_minus1)
		
		#Get leading, subleading and fourth leading taus
		leading_tau = tau[:,0]
		subleading_tau = tau[:,1]
		thirdleading_tau = tau[:,2]
		fourthleading_tau = tau[:,3]
		
		#Fill plots
		pt_hist.fill(leading_tau.pt)
		eta_hist.fill(leading_tau.eta)
		phi_hist.fill(leading_tau.phi)
		pt_all_hist.fill("Leading",leading_tau.pt)
		pt_all_hist.fill("Subleading",subleading_tau.pt)
		pt_all_hist.fill("Third-leading",thirdleading_tau.pt)
		pt_all_hist.fill("Fourth-leading",fourthleading_tau.pt)
		pt4_hist.fill(fourthleading_tau.pt)
		
		#Ditau mass plots 
		dimass_all_hist.fill("Leading pair", ak.ravel(mass(tau_plus1[(deltaR11 < deltaR21)], tau_minus1[(deltaR11 < deltaR21)])))
		dimass_all_hist.fill("Leading pair", ak.ravel(mass(tau_plus2[(deltaR21 < deltaR11)], tau_minus1[(deltaR21 < deltaR11)])))
		dimass_all_hist.fill("Subleading pair", ak.ravel(mass(tau_plus1[(deltaR12 < deltaR22)], tau_minus2[(deltaR12 < deltaR22)])))
		dimass_all_hist.fill("Subleading pair", ak.ravel(mass(tau_plus2[(deltaR22 < deltaR12)], tau_minus2[(deltaR22 < deltaR12)])))
		dimass_all_hist *= 1/(dimass_all_hist.sum())		
	
		ditau_mass1_hist.fill(ak.ravel(mass(tau_plus1[(deltaR11 < deltaR21)], tau_minus1[(deltaR11 < deltaR21)])))	
		ditau_mass1_hist.fill(ak.ravel(mass(tau_plus1[(deltaR21 < deltaR11)], tau_minus2[(deltaR21 < deltaR11)])))
		ditau_mass1_hist *= (1/ditau_mass1_hist.sum())
		ditau_mass2_hist.fill(ak.ravel(mass(tau_plus2[(deltaR22 < deltaR12)], tau_minus2[(deltaR22 < deltaR12)])))	
		ditau_mass2_hist.fill(ak.ravel(mass(tau_plus2[(deltaR12 < deltaR22)], tau_minus1[(deltaR12 < deltaR22)])))
		ditau_mass2_hist *= (1/ditau_mass2_hist.sum())	

		return{
			dataset: {
				"entries" : len(events),
				"pT": pt_hist,
				"eta":eta_hist,
				"phi":phi_hist,
				"pT_all": pt_all_hist,
				"pT_4": pt4_hist,
				"mass1": ditau_mass1_hist,
				"mass2": ditau_mass2_hist,
				"ditau_mass": dimass_all_hist
			}
		}	
	
	def postprocess(self, accumulator):
		pass	


if __name__ == "__main__":
	mass_str_arr = ["1000","2000","3000"]
	#mass_str_arr = ["2000"]
	trigger_bit_list = [40]
	#trigger_dict = {"PFHT500_PFMET100_PFMHT100_IDTight": [39], "AK8PFJet400_TrimMass30": [40]}

	tau_hist_dict = {
		"pT" :["leading_PtPlot", r"Leading $\tau$ $p_T$"], "eta": ["leading_etaPlot", r"Leading $\tau$ $\eta$"], "phi": ["leading_phiPlot", r"Leading $\tau$ $\phi$"], 
		"pT_all": ["AllPt_Plot",r"$4-\tau$ event transverse momenta"], "mass1":  ["Ditau_Mass1",r"Leading Di-$\tau$ pair mass"], "mass2": ["Ditau_Mass2",r"Subleading Di-$\tau$ pair mass"], 
		"ditau_mass": ["AllDitauMass_Plot",r"Di-$\tau$ pair masses"]
	}
	
	trigger_AK8Jet_hist_dict_1d = {
		"AK8JetSoftMass_Trigg" : ["AK8SoftMass_Trigger_Plot","AK8SoftDrop Mass Trigger"] , "AK8JetSoftMass_PreTrigg" : ["AK8SoftMass_NoTrigger_Plot","AK8SoftDrop Mass No Trigger"], 
		"AK8JetPt_Trigg" : ["AK8Pt_Trigger_Plot",r"AK8Jet $p_T$ Trigger"], "AK8JetPt_PreTrigg" : ["AK8Pt_NoTrigger_Plot",r"AK8Jet $p_T$ No Trigger"]
	}
	
	trigger_AK8Jet_hist_dict_2d = {
		"AK8Jet_PreTrigg" : ["AK8Jet_PreTriggerHist_Plot", "AK8Jet 2D Histogram No Trigger"], "AK8Jet_Trigg" : ["AK8Jet_TriggerHist_Plot", "AK8Jet 2D Histogram Trigger"],
		"AK8Jet_eff" : ["AK8Jet_Eff_Plot", "AK8Jet 2D Efficiency Histogram Trigger"]
	}

	trigger_MTHTJet_hist_dict_1d = {
		"MET_Trigg" : ["MET_Trigger_Plot","pfMET Trigger"] , "MET_PreTrigg" : ["MET_NoTrigger_Plot","pfMET No Trigger"], 
		"HT_Trigg" : ["HT_Trigger_Plot",r"HT Trigger"], "HT_PreTrigg" : ["HT_NoTrigger_Plot", r"HT No Trigger"]
	}
	
	trigger_MTHTJet_hist_dict_2d = {
		"Jet_PreTrigg" : ["Jet_PreTriggerHist_Plot", "MET and HT 2D Histogram No Trigger"], "Jet_Trigg" : ["Jet_TriggerHist_Plot", "MET and HT 2D Histogram Trigger"],
		"Jet_eff" : ["Jet_Eff_Plot", "MET and HT Efficiency Histogram Trigger"]
	}
	
	trigger_dict = {"PFHT500_PFMET100_PFMHT100_IDTight": 39, "AK8PFJet400_TrimMass30": 40}

	filebase = "~/Analysis/BoostedTau/TriggerEff/2018_Samples/GluGluToRadionToHHTo4T_M-"
	
	#Signal
	for mass_str in mass_str_arr:
		fileName = filebase + mass_str + ".root"
		events = NanoEventsFactory.from_root(
			fileName,
			treepath="/4tau_tree",
			schemaclass = BaseSchema,
			metadata={"dataset": "boosted_tau"},
		).events()
		p = TauPlotting()
		out = p.process(events)
		#Tau plotting
		for var_name, hist_name_arr in tau_hist_dict.items():
			fig, ax = plt.subplots()
			out["boosted_tau"][var_name].plot1d(ax=ax)
			plt.title(hist_name_arr[1], wrap=True)
			if (hist_name_arr[0] == "AllDitauMass_Plot"):
				ax.legend(title=r"Di-$\tau$ Pair")
			plt.savefig(hist_name_arr[0] + "-" + mass_str)
			plt.close()

		#Trigger Plotting
		print("Mass: " + mass_str[0] + "." + mass_str[1] + " TeV")
		for trigger_name, trigger_bit in trigger_dict.items():
			p2 = TriggerStudies(trigger_bit)
			trigger_out = p2.process(events)
			if (trigger_bit == 40):
				trigger_hist_dict_1d = trigger_AK8Jet_hist_dict_1d 
				trigger_hist_dict_2d = trigger_AK8Jet_hist_dict_2d 
			if (trigger_bit == 39):
				trigger_hist_dict_1d = trigger_MTHTJet_hist_dict_1d  
				trigger_hist_dict_2d = trigger_MTHTJet_hist_dict_2d 
			
			for var_name, hist_name_arr in trigger_hist_dict_1d.items():
				fig, ax = plt.subplots()
				trigger_out["boosted_tau"][var_name].plot1d(ax=ax)

				if (hist_name_arr[0][-14:] == "NoTrigger_Plot"):
					plt.title(hist_name_arr[1] + " mass : " + mass_str[0] + " TeV", wrap=True)
				else:
					plt.title(hist_name_arr[1] + " (" + trigger_name + ") , mass : " + mass_str[0] + " TeV", wrap=True)
				plt.savefig(hist_name_arr[0] + "-" + mass_str + "-" + trigger_name)
				plt.close()
				  
			for var_name, hist_name_arr in trigger_hist_dict_2d.items():
				fig, ax = plt.subplots()
				trigger_out["boosted_tau"][var_name].plot2d(ax=ax)

				if (hist_name_arr[0][-19:] == "PreTriggerHist_Plot"):
					plt.title(hist_name_arr[1] + " mass : " + mass_str[0] + " TeV", wrap=True)
				else:
					plt.title(hist_name_arr[1] + " (" +  trigger_name + "), mass : " + mass_str[0] + " TeV", wrap=True)
				plt.savefig(hist_name_arr[0] + "-" + mass_str + "-" + trigger_name)
				plt.close()
			
	#Obtain background information
	#background_array = ["ZZ4l",]
	file_base = "~/Analysis/BoostedTau/TriggerEff/2018_Background/"
	background_dict = {"ZZ4l" : r"$ZZ \rightarrow 4l$", "top": "Top Background"}
	#file_dict = {"ZZ4l": [file_base + "ZZ4l.root"], "top": [file_base + "Tbar-tchan.root",file_base + "Tbar-tW.root",file_base + "T-tchan.root"]}
	file_dict = {"top": [file_base + "Tbar-tchan.root",file_base + "Tbar-tW.root",file_base + "T-tchan.root"]}

	iterative_runner = processor.Runner(
		executor = processor.IterativeExecutor(compression=None),
		schema=BaseSchema
	)

	for background_name, title in background_dict.items():
		if (background_name == "ZZ4l"):
			events = NanoEventsFactory.from_root(
				"~/Analysis/BoostedTau/TriggerEff/2018_Background/" + background_name + ".root",
				treepath="/4tau_tree",
				schemaclass = BaseSchema,
				metadata={"dataset": "boosted_tau"},
			).events()

		
		#p2 = iterative_runner(file_dict[background_name], treename="4tau_tree",processor_instance=TriggerStudies())
		#p2 = TriggerStudies()
		print("Background: " + background_name)	
		for trigger_name, trigger_bit in trigger_dict.items():
			if (background_name == "top" and trigger_bit == 40):
				continue
			if (background_name == "ZZ4l"):
				p2 = TriggerStudies(trigger_bit, False)
				trigger_out = p2.process(events)
			else:
				trigger_out = iterative_runner(file_dict, treename="mutau_tree",processor_instance=TriggerStudies(trigger_bit, False)) 
			
			if (trigger_bit == 40):
				trigger_hist_dict_1d = trigger_AK8Jet_hist_dict_1d 
				trigger_hist_dict_2d = trigger_AK8Jet_hist_dict_2d 
			
			if (trigger_bit == 39):
				trigger_hist_dict_1d = trigger_MTHTJet_hist_dict_1d  
				trigger_hist_dict_2d = trigger_MTHTJet_hist_dict_2d 
			
			for var_name, hist_name_arr in trigger_hist_dict_1d.items():
				fig, ax = plt.subplots()
				if (background_name == "ZZ4l"):
					trigger_out["boosted_tau"][var_name].plot1d(ax=ax)
				else:
					#trigger_out[background_name]["boosted_tau"][var_name].plot1d(ax=ax)
					trigger_out[background_name][var_name].plot1d(ax=ax)
	
				if (hist_name_arr[0][-14:] == "NoTrigger_Plot"):
					plt.title(hist_name_arr[1] + title, wrap=True)
				else:
					plt.title(hist_name_arr[1] + " (" + trigger_name + r"), " + title, wrap=True)
				plt.savefig(hist_name_arr[0] + "-" + background_name + "-" + trigger_name)
				plt.close()
				  
			for var_name, hist_name_arr in trigger_hist_dict_2d.items():
				fig, ax = plt.subplots()
				if (background_name == "ZZ4l"):
					trigger_out["boosted_tau"][var_name].plot2d(ax=ax)
				else:
					#trigger_out[background_name]["boosted_tau"][var_name].plot2d(ax=ax)
					trigger_out[background_name][var_name].plot2d(ax=ax)
	
				if (hist_name_arr[0][-19:] == "PreTriggerHist_Plot"):
					plt.title(hist_name_arr[1] + title, wrap=True)
				else:
					plt.title(hist_name_arr[1] + trigger_name + "), " + title, wrap=True)
				plt.savefig(hist_name_arr[0] + "-" + background_name + "-" + trigger_name)
				plt.close()
		
