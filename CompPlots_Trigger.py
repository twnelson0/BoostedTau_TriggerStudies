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

def bit_mask(in_bits):
	mask = 0
	for bit in in_bits:
		mask += (1 << bit)
	return mask


class TriggerStudies(processor.ProcessorABC):
	def __init__(self, trigger_bit, trigger_cut = True, offline_cut = False, signal = True):
		self.trigger_bit = trigger_bit
		self.signal = signal
		self.trigger_cut = trigger_cut
		self.offline_cut = offline_cut
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

				#for x,y in zip(ak.ravel(Muon.nMu), ak.ravel(Electron.nEle)):
				#	if (x > 0):
				#		print("%d muons"%x)
				#	if (y > 0):
				#		print("%d electrons"%y)
				#	print("%d leptons"%(x + y))
				
					
				#AK8Jet = AK8Jet[AK8Jet.nMu == 0]	
				#print("%d Events"%len(ak.ravel(AK8Jet.nMu)))
				#AK8Jet = AK8Jet[AK8Jet.nEle == 0]	
				#print("%d Events"%len(ak.ravel(AK8Jet.nMu)))
			AK8Pt_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetPt))
			AK8Pt_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetPt)
			AK8SoftMass_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetDropMass))
			AK8SoftMass_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetDropMass)
			#AK8Pt_all.fill("No Trigger",ak.ravel(AK8Jet.AK8JetPt))
		
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
		
		#Delta R selection
		if (self.signal):
			#print(ak.ravel(deltaR11 < deltaR12 and deltaR11 < deltaR21))
			#Store count information (they're all the same)
			counts = ak.num(deltaR11)
			
			deltaR11 = np.array(ak.flatten(deltaR11))
			deltaR12 = np.array(ak.flatten(deltaR12))
			deltaR21 = np.array(ak.flatten(deltaR21))
			deltaR22 = np.array(ak.flatten(deltaR22))
			
			#print(deltaR11)
			#print(deltaR12)
			#print(deltaR21)

			deltaRDoubletCut = ((deltaR11 < deltaR12).all() and (deltaR11 < deltaR21).all() and (deltaR11 < 0.8).all()) ^ ((deltaR12 < deltaR11).all() and (deltaR12 < deltaR22).all() and (deltaR12 < 0.8).all())
			deltaRDoubletCut = ak.unflatten(ak.from_numpy(deltaRDoubletCut, counts))
			print("Delta R Cut obtained")
			tau = tau[deltaRDoubletCut] 
			
			#print(deltaR11)
			#print(deltaR11 < deltaR12)
			#print(deltaR11 < deltaR21)
			#tau = tau[]
		
		#Efficiency Histograms (How do I do these??)
		if (self.trigger_bit == 40):	
			AK8Pt_Trigg.fill(ak.ravel(AK8Jet.AK8JetPt))
			AK8SoftMass_Trigg.fill(ak.ravel(AK8Jet.AK8JetDropMass))
			#AK8Pt_all.fill("Trigger",ak.ravel(AK8Jet.AK8JetPt))	
			AK8Pt_Trigg_Arr = ak.ravel(AK8Jet.AK8JetPt)
			#AK8SoftMass_all.fill("Trigger",ak.ravel(AK8Jet.AK8JetDropMass))	
			AK8SoftMass_Trigg_Arr = ak.ravel(AK8Jet.AK8JetDropMass)
			pre_triggernum = ak.num(AK8Pt_NoTrigg_Arr,axis=0)
			post_triggernum = ak.num(AK8Pt_Trigg_Arr,axis=0)
            
			if (self.signal):
				print("Efficiency (AK8Jet Trigger): %f"%(ak.num(AK8Pt_Trigg_Arr,axis=0)/ak.num(AK8Pt_NoTrigg_Arr,axis=0)))
			AK8Jet_PreTrigger.fill(AK8Pt_NoTrigg_Arr, AK8SoftMass_NoTrigg_Arr)
			AK8Jet_Trigger.fill(AK8Pt_Trigg_Arr, AK8SoftMass_Trigg_Arr)
			eff_AK8Jet = AK8Jet_Trigger/AK8Jet_PreTrigger
		
		if (self.trigger_bit == 39):
			HT_Trigg.fill(ak.ravel(Jet.HT))
			HT_Trigg_Arr = ak.ravel(Jet.HT)
			MET_Trigg.fill(ak.ravel(Jet.pfMET))
			MET_Trigg_Arr = ak.ravel(Jet.pfMET)
			pre_triggernum = ak.num(MET_NoTrigg_Arr,axis=0)
			post_triggernum = ak.num(MET_Trigg_Arr,axis=0)	
			
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
					"pre_trigger_num": pre_triggernum, 
					"post_trigger_num": post_triggernum,
					"AK8Pt_NoTrigg_Arr": AK8Pt_NoTrigg_Arr,
					"AK8SoftMass_NoTrigg_Arr": AK8Pt_Trigg_Arr 
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
					"pre_trigger_num": pre_triggernum,
					"post_trigger_num": post_triggernum

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

	signal_base = "~/Analysis/BoostedTau/TriggerEff/2018_Samples/GluGluToRadionToHHTo4T_M-"
	background_base = "~/Analysis/BoostedTau/TriggerEff/2018_Background/"
	file_dict = {"1 TeV": [signal_base + "1000.root"], "2 TeV": [signal_base + "2000.root"], "ZZ4l": [background_base + "ZZ4l.root"]}

	iterative_runner = processor.Runner(
		executor = processor.IterativeExecutor(compression=None),
		schema=BaseSchema
	)


	#Obtain background information
	#background_array = ["ZZ4l",]
	file_base = "~/Analysis/BoostedTau/TriggerEff/2018_Background/"
	#background_dict = {"ZZ4l" : r"$ZZ \rightarrow 4l$", "top": "Top Background"}
	#file_dict = {"ZZ4l": [file_base + "ZZ4l.root"], "top": [file_base + "Tbar-tchan.root",file_base + "Tbar-tW.root",file_base + "T-tchan.root"]}
	#file_dict = {"top": [file_base + "Tbar-tchan.root",file_base + "Tbar-tW.root",file_base + "T-tchan.root"]}

	
	#for x in legend_arr:
	#	print(trigger_out[x]["AK8Pt_NoTrigg_Arr"])
			
	title_arr = [r"AK8 Jet $p_T$ (Before Trigger)",r"AK8 Jet $p_T$ (After Trigger)",r"AK8 Jet Soft Mass (Before Trigger)",r"AK8 Jet Soft Mass (After Trigger)",
				r"MET (Before Trigger)",r"MET (After Trigger)",r"HT (Before Trigger)",r"HT (After Trigger)"]
	plot_arr = ["AK8JetPt_PreTrigg","AK8JetPt_Trigg","AK8JetSoftMass_PreTrigg","AK8JetSoftMass_Trigg","MET_PreTrigg","MET_Trigg","HT_PreTrigg","HT_Trigg"]
	sample_arr = ["1 TeV","2 TeV", "ZZ4l"]	
	trigger_arr = [39,40]
	
	#for plot,title in zip(plot_arr, title_arr):
	for trigger in trigger_arr:
		trigger_out = iterative_runner(file_dict, treename="4tau_tree",processor_instance=TriggerStudies(trigger,signal = True)) 
		for i in range(8):
			fig, ax = plt.subplots()
			if (np.floor(i/4) == trigger % 2):
				#Construct dictionary
				trigger_dict = {}
				for sample in sample_arr:
					trigger_dict[sample] = trigger_out[sample][plot_arr[i]]*(1/trigger_out[sample][plot_arr[i]].sum())
		
				stack = hist.Stack.from_dict(trigger_dict)
				stack.plot()
				plt.title(title_arr[i])
				plt.legend()
				plt.savefig(plot_arr[i] + "_Comp_Plot")
				plt.close()
			
	
	#fig, ax = plt.subplots()
	#trigger_dict = {"1 TeV": trigger_out["1 TeV"]["AK8JetPt_PreTrigg"], "2 TeV": trigger_out["2 TeV"]["AK8JetPt_PreTrigg"], "ZZ4l": trigger_out["ZZ4l"]["AK8JetPt_PreTrigg"]}
	#AK8Pt Plots
	#stack = hist.Stack.from_dict(trigger_dict)
	#stack.plot()
	#plt.title(r"AK8 $p_T$ (Before Trigger)")
	#plt.legend()
	#plt.savefig("AK8PT_Comp_PreTrigger")		
	#plt.close()
	
	#stack = hist.Stack.from_dict({"1 TeV": trigger_out["1 TeV"]["AK8JetPt_Trigg"], "2 TeV": trigger_out["2 TeV"]["AK8JetPt_Trigg"], "ZZ4l": trigger_out["ZZ4l"]["AK8JetPt_Trigg"]})
	#stack.plot()
	#plt.title(r"AK8 $p_T$ (After Trigger)")
	#plt.legend()
	#plt.savefig("AK8PT_Comp_Trigger")		
	#plt.close()
	
	#AK8SoftMass
		
