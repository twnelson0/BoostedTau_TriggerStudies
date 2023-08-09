import awkward as ak
import uproot
import hist
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
from coffea import processor, nanoevents
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import candidate
from math import pi

def deltaR(tau1, tau2):
	return np.sqrt((tau2.eta - tau1.eta)**2 + (tau2.phi - tau1.phi)**2)

def mass(tau1,tau2):
	return np.sqrt((tau1.E + tau2.E)**2 - (tau1.Px + tau2.Px)**2 - (tau1.Py + tau2.Py)**2 - (tau1.Pz + tau2.Pz)**2)

#Evaluate trigger selection
def trigger_selector(trigger_bit_list, input_trigger):
	trigger_bin = bin(input_trigger)[2:] #Get binary rerpresntation of trigger
	max_bits = len(trigger_bin) - 1 #Get largest trigger bit
	pass_selection = True
	
	for search_bit in trigger_bit_list:
		if (max_bits < search_bit):
			pass_selection = False
			break
		else:
			if (trigger_bin[max_bits - search_bit] == '1'):
				continue
			if (trigger_bin[max_bits - search_bit] == '0'):
				pass_selection = False
				break
	
	return pass_selection

class TriggerStudies(processor.ProcessorABC):
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
				"trigger": events.HLTJet,
				"MET": events.pfMET,
			},
			with_name="TauArray",
			behavior=candidate.behavior,
		)
		
		#Histograms
		MET_hist = (
			hist.Hist.new
            .Reg(50, 0, 1500., name="p_T", label="$\slashed{E}$ [GeV]") 
            .Int64()
		)
	
		#Tau Selection
		print("================No Cuts================")
		#for x in tau:
		#	print(np.log2(x.trigger))
		#foo_arr = ak.sum(np.log2(tau.trigger), axis=1)/ak.num(tau)
		#print(ak.sum(np.log2(tau.trigger), axis=1)/ak.num(tau) == np.log2(ak.sum(tau.trigger, axis=1)/ak.num(tau)))
		#num_evnt = 0
		#for x in foo_arr:
		#	if (x == 39):
		#		num_evnt += 1
		#for x,y in zip(np.log2(ak.sum(tau.trigger, axis=1)/ak.num(tau)),tau):
		#	print(np.log2(y.trigger))
		#	print("Raw Trigger:")
		#	print(y.trigger)
		#	print(x)		

		#tau = tau[np.log2(ak.sum(tau.trigger, axis=1)/ak.num(tau)) == 39] #Trigger selection
		tau = tau[trigger_selector([39],tau.trigger[0])] #Trigger selection
		#print("================Trigger Selection================")
		#for x in tau:
		#	print(np.log2(x.trigger))
		#	print("MET:")
		#	print(x.MET)
		#tau = tau[tau.pt > 30] #pT
		#tau = tau[tau.eta < 2.3] #eta
		#print("================pT and eta cuts================")
		#for x in tau:
		#	print(x.MET)
		
		#Loose isolation
		tau = tau[tau.iso1 >= 0.5]
		tau = tau[tau.iso2 >= 0.5]		
		#print("================Isolation cuts================")
		#for x in tau:
		#	print(x.MET)
		
		tau = tau[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		tau = tau[ak.num(tau) == 4] #4 tau events 	
		#print("================4 tau and charge conservation cuts================")
		#for x in tau:
		#	print(x.MET)
		#	print(np.log2(x.trigger))

		#print("================Trigger cut================")
		#for x in tau:
		#	print(x.MET)
		#	print(x.trigger)
		tau_plus = tau[tau.charge > 0]	
		tau_minus = tau[tau.charge < 0]

		#Construct all possible valid ditau pairs
		tau_plus1, tau_plus2 = ak.unzip(ak.combinations(tau_plus,2))
		tau_minus1, tau_minus2 = ak.unzip(ak.combinations(tau_minus,2))
		
		deltaR11 = deltaR(tau_plus1, tau_minus1)
		deltaR12 = deltaR(tau_plus1, tau_minus2)
		deltaR22 = deltaR(tau_plus2, tau_minus2)
		deltaR21 = deltaR(tau_plus2, tau_minus1)

		#Fill Histograms
		#print(tau.MET)
		for x in tau:
			print(x.MET)
		MET_hist.fill(tau.MET)

		return{
			 dataset: {
				"MET": MET_hist,
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
				#"iso": events.boostedTauByVLooseIsolationMVArun2v1DBoldDMwLTNew,
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
			#.Int64()
		)
			
		#Apply cuts/selection
		tau = tau[tau.pt > 30] #pT
		#print("=====================Pre-Eta Cut=====================")
		#for test in tau:
		#	if (len(test.pt) > 4 and sum(test.charge) == 0):
		#		print("Event should be counted?")
		#		print(test.eta)
		#		print(test.charge)
		tau = tau[tau.eta < 2.3] #eta
		#print("=====================Post-Eta Cut=====================")
		#for test in tau:
		#	if (len(test.pt) > 4 and sum(test.charge) != 0):
		#		print("Event should be counted?")
		#		print(test.eta)
		#		print(test.charge)

		
		#Loose isolation
		tau = tau[tau.iso1 >= 0.5]
		tau = tau[tau.iso2 >= 0.5]		
		
		tau = tau[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		
		print("Before 4 tau cut length is: %d" % len(tau))
		tau = tau[ak.num(tau) == 4] #4 tau events (unsure about this)	
		print("After 4 tau cut length is: %d" % len(tau))
		#for i in range(10):
		#	print(tau[i].pt)
		tau_plus = tau[tau.charge > 0]	
		tau_minus = tau[tau.charge < 0]

		#Construct all possible valid ditau pairs
		tau_plus1, tau_plus2 = ak.unzip(ak.combinations(tau_plus,2))
		tau_minus1, tau_minus2 = ak.unzip(ak.combinations(tau_minus,2))
		
		#Sanity check on how events are unpacked
		#for tau1, tau2 in zip(tau_plus1, tau_plus2):
		#	if (tau1.pt < tau2.pt):
		#		print("Things don't work the way I think they do!!")
		#for tau1, tau2 in zip(tau_minus1, tau_minus2):
		#	if (tau1.pt < tau2.pt):
		#		print("Things don't work the way I think they do!!")
		
		deltaR11 = deltaR(tau_plus1, tau_minus1)
		deltaR12 = deltaR(tau_plus1, tau_minus2)
		deltaR22 = deltaR(tau_plus2, tau_minus2)
		deltaR21 = deltaR(tau_plus2, tau_minus1)
	
		#Old Pairing
		#pairing_11 = (deltaR11 < deltaR12) & (deltaR11 < deltaR21) & (deltaR11 < deltaR22)
		#pairing_12 = (deltaR12 < deltaR11) & (deltaR12 < deltaR21) & (deltaR12 < deltaR22)
		#pairing_21 = (deltaR21 < deltaR11) & (deltaR21 < deltaR12) & (deltaR21 < deltaR22)
		#pairing_22 = (deltaR22 < deltaR12) & (deltaR22 < deltaR21) & (deltaR22 < deltaR11)
		
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
		
		#Ditau mass plots (I think all my cuts are fine, maybe with the exception of the pairing indicies and the indifference of pT)
		#dimass_all_hist.fill("Pair 1", ak.ravel(mass(tau_plus1[pairing_11], tau_minus1[pairing_11])))
		#dimass_all_hist.fill("Pair 1", ak.ravel(mass(tau_plus2[pairing_21], tau_minus2[pairing_21])))
		#dimass_all_hist.fill("Pair 2", ak.ravel(mass(tau_plus1[pairing_22], tau_minus2[pairing_22])))
		#dimass_all_hist.fill("Pair 2", ak.ravel(mass(tau_plus2[pairing_12], tau_minus1[pairing_12])))
		
		#Ditau mass plots 
		dimass_all_hist.fill("Leading pair", ak.ravel(mass(tau_plus1[(deltaR11 < deltaR21)], tau_minus1[(deltaR11 < deltaR21)])))
		dimass_all_hist.fill("Leading pair", ak.ravel(mass(tau_plus2[(deltaR21 < deltaR11)], tau_minus1[(deltaR21 < deltaR11)])))
		dimass_all_hist.fill("Subleading pair", ak.ravel(mass(tau_plus1[(deltaR12 < deltaR22)], tau_minus2[(deltaR12 < deltaR22)])))
		dimass_all_hist.fill("Subleading pair", ak.ravel(mass(tau_plus2[(deltaR22 < deltaR12)], tau_minus2[(deltaR22 < deltaR12)])))
		dimass_all_hist *= 1/(dimass_all_hist.sum())		
	
		#Normalization scales (currently broken/wrong)		
		#scale_leading = ak.sum(ak.ravel(mass(tau_plus1[(deltaR11 < deltaR21)], tau_minus1[(deltaR11 < deltaR21)]))) + ak.sum(ak.ravel(mass(tau_plus1[(deltaR21 < deltaR11)], tau_minus2[(deltaR21 < deltaR11)])))
		#scale_subleading = ak.sum(ak.ravel(mass(tau_plus2[(deltaR22 < deltaR12)], tau_minus2[(deltaR22 < deltaR12)]))) + ak.sum(ak.ravel(mass(tau_plus2[(deltaR12 < deltaR22)], tau_minus1[(deltaR12 < deltaR22)])))		

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
	#mass_str_arr = ["1000","2000","3000"]
	mass_str_arr = ["2000"]
	#filebase_arr = [""]
	filebase = "~/Analysis/BoostedTau/TriggerEff/2018_Samples/GluGluToRadionToHHTo4T_M-"
	
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
		fig, ax = plt.subplots()
		out["boosted_tau"]["pT"].plot1d(ax=ax)
		plt.title(r"Leading $\tau$ $p_T$")
		plt.savefig("leading_PtPlot-" + mass_str)
		plt.cla()
		out["boosted_tau"]["eta"].plot1d(ax=ax)
		plt.title(r"Leading $\tau$ $\eta$")
		plt.savefig("leading_etaPlot-" + mass_str)
		plt.cla()
		out["boosted_tau"]["phi"].plot1d(ax=ax)
		plt.title(r"Leading $\tau$ $\phi$")
		plt.savefig("leading_phiPlot-" + mass_str)
		plt.cla()
		out["boosted_tau"]["pT_all"].plot1d(ax=ax)
		plt.title(r"$4-\tau$ event transverse momenta")
		ax.legend(title=r"$\tau$")
		plt.savefig("AllPt_Plot-" + mass_str)
		plt.cla()
		out["boosted_tau"]["pT_4"].plot1d(ax=ax)
		plt.title(r"Fourth leading $\tau$ $p_T$")
		plt.savefig("FourthLeadingPt-" + mass_str)
		plt.cla()
		out["boosted_tau"]["mass1"].plot1d(ax=ax)
		plt.title(r"Leading Di-$\tau$ pair mass")
		plt.savefig("Ditau_Mass1-" + mass_str)
		plt.cla()
		out["boosted_tau"]["mass2"].plot1d(ax=ax)
		plt.title(r"Subleading Di-$\tau$ pair mass")
		plt.savefig("Ditau_Mass2-" + mass_str)
		plt.cla()
		out["boosted_tau"]["ditau_mass"].plot1d(ax=ax)
		plt.title(r"Di-$\tau$ pair masses")
		ax.legend(title=r"Di-$\tau$ Pair")
		plt.savefig("AllDitauMass_Plot-" + mass_str)
		plt.cla()		

		p2 = TriggerStudies()
		trigger_out = p2.process(events)
		trigger_out["boosted_tau"]["MET"].plot1d(ax=ax)
		plt.title(r"$\slashed{E}$ after trigger cut")
		plt.savefig("MET_Trigger_Plot-" + mass_str)


