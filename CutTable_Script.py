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

#def MHT(jet_evnt, jet_up_evnt, jet_down_evnt):
	#MHT_Val = 0
	
	#for jet, up, down in zip(jet_evnt, jet_up_evnt, jet_down_evnt):
		#MHT_Val += 

def bit_mask(in_bits):
	mask = 0
	for bit in in_bits:
		mask += (1 << bit)
	return mask

def DrawTable(table_title, table_name, table_dict):
	file_name = "Efficiency_Table_" + table_name + ".tex"
	file = open(file_name,"w")

	#Set up the Tex Document
	file.write("\\documentclass{article} \n")
	file.write("\\usepackage{multirow} \n")
	file.write("\\usepackage{multirow} \n")
	#file.write("\\usepackage{lscape}\n")
	#file.write("\\usepacke")
	file.write("\\begin{document} \n")
	file.write("\\centering \n")
	
	#Set up the table
	file.write("\\begin{tabular}{|p{3cm}|p{6cm}|p{5cm}|} \n")
	file.write("\\hline \n")
	file.write("\\multicolumn{3}{|c|}{" + table_title  + "} \\\\ \n")
	file.write("\\hline \n")
	file.write("Sample File(s) & PFMET500\\_PFMET100 PFMHT100\\_IDTight Efficiency & AK8PFJet400\\_TrimMass30 Efficiency \\\\ \n")
	file.write("\\hline \n")
	
	#Fill table
	for sample, eff_arr in table_dict.items():
		file.write(sample + " & " + str(eff_arr[0]) + " & " + str(eff_arr[1]) + "\\\\")
		file.write("\n")
	file.write("\\hline \n")
	file.write("\\end{tabular}")
	file.write("\\end{document}")
	file.close()

	

#class EffTable()

class TriggerStudies(processor.ProcessorABC):
	def __init__(self, trigger_bit, trigger_cut = True, offline_cut = False, signal = True):
		self.trigger_bit = trigger_bit
		self.signal = signal
		self.offline_cut = offline_cut
		self.trigger_cut = trigger_cut
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
				"Pt": events.jetPt,
				"pfMET": events.pfMET,
				#"MHT_x": events.jetPt*0, 
				#"MHT_y": events.jetPt*0, 
				#"MHT": events.jetPt*0, 
				"PtTotUncDown": events.jetPtTotUncDown,
				"PtTotUncUp": events.jetPtTotUncUp,
				"PFLooseId": events.jetPFLooseId,
				"HT": ak.sum(events.jetPt, axis=1),
				"eta": events.jetEta,
				"phi": events.jetPhi,
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
		Muon = Muon[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		Electron = Electron[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		Jet = Jet[(ak.sum(tau.charge,axis=1) == 0)]
		tau = tau[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation


		AK8Jet = AK8Jet[ak.num(tau) == 4]
		Electron = Electron[ak.num(tau) == 4]
		Muon = Muon[ak.num(tau) == 4]
		Jet = Jet[ak.num(tau) == 4]
		#print(len(tau[ak.num(tau) > 4]))
		tau = tau[ak.num(tau) == 4] #4 tau events
		
		#Set up variables for offline cuts
		#MHT
		Jet_MHT = Jet[Jet.Pt > 30]
		Jet_MHT = Jet_MHT[np.abs(Jet_MHT.eta) < 5]
		Jet_MHT = Jet_MHT[Jet_MHT.PFLooseId > 0.5]
		JetUp_MHT = Jet[Jet.PtTotUncUp > 30]
		JetUp_MHT = JetUp_MHT[np.abs(JetUp_MHT.eta) < 5]
		JetUp_MHT = JetUp_MHT[JetUp_MHT.PFLooseId > 0.5]
		JetDown_MHT = Jet[Jet.PtTotUncDown > 30]
		JetDown_MHT = JetDown_MHT[np.abs(JetDown_MHT.eta) < 5]
		JetDown_MHT = JetDown_MHT[JetDown_MHT.PFLooseId > 0.5]
		#print(len(Jet_MHT))
		#print(len(Jet))
		#print(len(JetUp_MHT))
		#print(len(JetDown_MHT))
		#print(Jet.Pt)
		#print(ak.sum(Jet_MHT.Pt*np.cos(Jet_MHT.phi),axis=1,keepdims=True))
		Jet["MHT_x"] = ak.sum(Jet_MHT.Pt*np.cos(Jet_MHT.phi),axis=1,keepdims=False) + ak.sum(JetUp_MHT.PtTotUncUp*np.cos(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.cos(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet["MHT_y"] = ak.sum(Jet_MHT.Pt*np.sin(Jet_MHT.phi),axis=1,keepdims=False) + ak.sum(JetUp_MHT.PtTotUncUp*np.sin(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.sin(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet["MHT"] = np.sqrt(Jet.MHT_x**2 + Jet.MHT_y**2)
		print("Jet MHT Defined:")
		#print(Jet.MHT)

		#HT
		tau_jet = ak.cartesian({"tau": tau, "Jet_MHT": Jet_MHT},axis=1)
		tau_jetUp = ak.cartesian({"tau": tau, "JetUp_MHT": JetUp_MHT},axis=1)
		tau_jetDown = ak.cartesian({"tau": tau, "JetDown_MHT": JetDown_MHT},axis=1)
		#print("Tau x Jet Object:")
		#print(deltaR(tau_jet["tau"],tau_jet["Jet_MHT"]))
		#print(len(deltaR(tau_jet["tau"],tau_jet["Jet_MHT"])))
		#print(len(Jet_MHT))
		Jet_MHT["dR"] = ak.prod(ak.unflatten(deltaR(tau_jet["tau"],tau_jet["Jet_MHT"]) >= 0.5,axis = 1, counts = 4), axis=2) #Clump jet and taus in structure
		JetUp_MHT["dR"] = ak.prod(ak.unflatten(deltaR(tau_jetUp["tau"],tau_jetUp["JetUp_MHT"]) >= 0.5, axis = 1, counts = 4), axis=2) 
		JetDown_MHT["dR"] = ak.prod(ak.unflatten(deltaR(tau_jetDown["tau"],tau_jetDown["JetDown_MHT"]) >= 0.5, axis = 1, counts = 4), axis=2) 
		#print(R_Cut)
		#print(len(R_Cut))

		#print("Len passing delta R Cut:"%len(Jet_MHT.dR >= 0.5))
		#print("Len Not passing delta R Cut:"%len(Jet_MHT.dR >= 0.5))
		Jet_HT = Jet_MHT[Jet_MHT.dR] #Lepton cuts
		JetUp_HT = JetUp_MHT[JetUp_MHT.dR]
		JetDown_HT = JetDown_MHT[JetDown_MHT.dR]
		#print("Post HT len = %d"%len(Jet_HT))
		Jet["HT"] = ak.sum(Jet_HT.Pt,axis = 1,keepdims=False) + ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=False) + ak.sum(JetDown_HT.PtTotUncDown,axis = 1,keepdims=False)
		#print("HT??")
		#print(Jet.HT)
		#print("R Len = %d"%len(R_Arr))

		if (self.trigger_bit == 40):
			
			AK8Pt_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetPt))
			AK8Pt_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetPt)
			AK8SoftMass_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetDropMass))
			AK8SoftMass_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetDropMass)
			AK8Pt_all.fill("No Trigger",ak.ravel(AK8Jet.AK8JetPt))
		
		if (self.trigger_bit == 39):
			#Apply Jet Cuts
			Jet = Jet[Jet.eta <= 3]	
			Jet = Jet[Jet.HT > 30]
			#print(Jet.HT)
			#print("MHT Callback:")
			#print(Jet.MHT)
			
			#Fill Histograms
			HT_PreTrigg.fill(ak.ravel(Jet.HT))
			HT_NoTrigg_Arr = ak.ravel(Jet.HT)
			MET_PreTrigg.fill(ak.ravel(Jet.pfMET))
			MET_NoTrigg_Arr = ak.ravel(Jet.pfMET)	

		if (self.trigger_cut):
			tau = tau[np.bitwise_and(tau.trigger,trigger_mask) == trigger_mask]
			AK8Jet = AK8Jet[np.bitwise_and(AK8Jet.trigger,trigger_mask) == trigger_mask]
			Jet = Jet[np.bitwise_and(Jet.trigger,trigger_mask) == trigger_mask]

		if (self.offline_cut):
			if (self.trigger_bit == 40):
				print("Offline Cut 40")
				tau = tau[ak.all(AK8Jet.AK8JetPt > 400, axis = 1)]
				Jet = Jet[ak.all(AK8Jet.AK8JetPt > 400, axis = 1)]
				AK8Jet = AK8Jet[ak.all(AK8Jet.AK8JetPt > 400, axis = 1)]
				tau = tau[ak.all(AK8Jet.AK8JetDropMass > 30, axis = 1)]
				Jet = Jet[ak.all(AK8Jet.AK8JetDropMass > 30, axis = 1)]
				AK8Jet = AK8Jet[ak.all(AK8Jet.AK8JetDropMass > 30, axis = 1)]
				print("Trigger Cuts applied to all")
			if (self.trigger_bit == 39):
				print("Offline Cut 39")
				#print(Jet.HT)
				#print(Jet.MHT)
				tau = tau[ak.all(Jet.PFLooseId, axis = 1)]
				AK8Jet = AK8Jet[ak.all(Jet.PFLooseId, axis = 1)]
				Jet = Jet[ak.all(Jet.PFLooseId, axis = 1)]
				print(len(tau))
				tau = tau[ak.all(Jet.MHT > 100, axis = 1)]
				AK8Jet = AK8Jet[ak.all(Jet.MHT > 100, axis = 1)]
				Jet = Jet[ak.all(Jet.MHT > 100, axis = 1)]
				print(len(tau))
				tau = tau[ak.all(Jet.HT > 500, axis = 1)]
				AK8Jet = AK8Jet[ak.all(Jet.HT > 500, axis = 1)]
				Jet = Jet[ak.all(Jet.HT > 500, axis = 1)]
				tau = tau[ak.all(Jet.pfMET > 100, axis = 1)]
				AK8Jet = AK8Jet[ak.all(Jet.pfMET > 100, axis = 1)]
				Jet = Jet[ak.all(Jet.pfMET > 100, axis = 1)]
				print("Trigger Cuts applied to all")
			
		tau_plus = tau[tau.charge > 0]
		#print(len(tau_plus))	
		tau_minus = tau[tau.charge < 0]
		#print(len(tau_minus))	

		#Construct all possible valid ditau pairs
		tau_plus1, tau_plus2 = ak.unzip(ak.combinations(tau_plus,2))
		tau_minus1, tau_minus2 = ak.unzip(ak.combinations(tau_minus,2))
		print(tau_plus1.eta)
		
		deltaR11 = deltaR(tau_plus1, tau_minus1)
		deltaR12 = deltaR(tau_plus1, tau_minus2)
		deltaR22 = deltaR(tau_plus2, tau_minus2)
		deltaR21 = deltaR(tau_plus2, tau_minus1)
		
		#if (self.signal):
		#	print(deltaR11 < deltaR21)	
		
		#Efficiency Histograms 
		if (self.trigger_bit == 40):	
			AK8Pt_Trigg.fill(ak.ravel(AK8Jet.AK8JetPt))
			AK8SoftMass_Trigg.fill(ak.ravel(AK8Jet.AK8JetDropMass))
			AK8Pt_all.fill("Trigger",ak.ravel(AK8Jet.AK8JetPt))	
			AK8Pt_Trigg_Arr = ak.ravel(AK8Jet.AK8JetPt)
			AK8SoftMass_all.fill("Trigger",ak.ravel(AK8Jet.AK8JetDropMass))	
			AK8SoftMass_Trigg_Arr = ak.ravel(AK8Jet.AK8JetDropMass)
			pre_triggernum = ak.num(AK8Pt_NoTrigg_Arr,axis=0)
			print("Number = %d"%pre_triggernum)
			post_triggernum = ak.num(AK8Pt_Trigg_Arr,axis=0)
			print("Number = %d"%post_triggernum)
            
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
			print("Number = %d"%pre_triggernum)
			post_triggernum = ak.num(MET_Trigg_Arr,axis=0)	
			print("Number = %d"%post_triggernum)
			
			if (self.signal):	
				print("Efficiency (HT+MET Trigger): %f"%(ak.num(MET_Trigg_Arr,axis=0)/ak.num(MET_NoTrigg_Arr,axis=0)))
			#eff_val_39 = (ak.num(MET_Trigg_Arr,axis=0)/ak.num(MET_NoTrigg_Arr,axis=0))
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
					"post_trigger_num": post_triggernum 
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
	table_dict = {}
	title_dict = {"1000": "1 TeV Signal", "2000": "2 TeV Signal", "3000": "3 TeV Siganl", "ZZ4l": r"\(ZZ \rightarrow 4l\)", "top": "Top Background"}
	file_base = "~/Analysis/BoostedTau/TriggerEff/2018_Background/"
	background_dict = {"ZZ4l" : r"$ZZ \rightarrow 4l$", "top": "Top Background"}
	#file_dict = {"ZZ4l": [file_base + "ZZ4l.root"], "top": [file_base + "Tbar-tchan.root",file_base + "Tbar-tW.root",file_base + "T-tchan.root"]}
	file_dict = {"top": [file_base + "Tbar-tchan.root",file_base + "Tbar-tW.root",file_base + "T-tchan.root"]}
	
	for i in range(3):
		if (i == 0):
			use_trigger = True
			use_offline= False
			table_title = "Trigger Efficiency Table"
			table_file_name = "Trigger_NoCuts"
		if (i == 1):
			use_trigger = False
			use_offline = True
			table_title = "Offline Cuts Efficiency Table"
			table_file_name = "NoTrigger_Cuts"
		if (i == 2):
			use_trigger = True
			use_offline = True
			table_title = "Trigger and Offline Cuts Efficiency Table"
			table_file_name = "Trigger_cuts"
	
		#Signal
		for mass_str in mass_str_arr:
			fileName = filebase + mass_str + ".root"
			events = NanoEventsFactory.from_root(
				fileName,
				treepath="/4tau_tree",
				schemaclass = BaseSchema,
				metadata={"dataset": "boosted_tau"},
			).events()
		
			#Tau plotting
			print("Mass: " + mass_str[0] + "." + mass_str[1] + " TeV")
			mass_eff_arr = [-1,-1]
			for trigger_name, trigger_bit in trigger_dict.items():
				p2 = TriggerStudies(trigger_bit, trigger_cut = use_trigger, offline_cut = use_offline, signal = True, )
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
				
					if (trigger_bit == 39):
						mass_eff_arr[0] = trigger_out["boosted_tau"]["post_trigger_num"]/trigger_out["boosted_tau"]["pre_trigger_num"]
					if (trigger_bit == 40):
						mass_eff_arr[1] = trigger_out["boosted_tau"]["post_trigger_num"]/trigger_out["boosted_tau"]["pre_trigger_num"]
			
				table_dict[title_dict[mass_str]] = mass_eff_arr #Update dictionary

		iterative_runner = processor.Runner(
			executor = processor.IterativeExecutor(compression=None),
			schema=BaseSchema
		)
		#Background 
		for background_name, title in background_dict.items():
			if (background_name == "top"): #Skip top background for now because it's extremely broken
				continue
			if (background_name == "ZZ4l"):
				events = NanoEventsFactory.from_root(
					"~/Analysis/BoostedTau/TriggerEff/2018_Background/" + background_name + ".root",
					treepath="/4tau_tree",
					schemaclass = BaseSchema,
					metadata={"dataset": "boosted_tau"},
				).events()
		
			print("Background: " + background_name)	
			eff_arr = [-1,-1]
			for trigger_name, trigger_bit in trigger_dict.items():
				if (background_name == "ZZ4l"):
					p2 = TriggerStudies(trigger_bit, trigger_cut = use_trigger, offline_cut = use_offline, signal = False)
					trigger_out = p2.process(events)
				else:
					trigger_out = iterative_runner(file_dict, treename="mutau_tree",processor_instance=TriggerStudies(trigger_bit, trigger_cut = use_trigger, offline_cut = use_offline, signal = False)) 
			
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
						print("Efficiency = %f"%(trigger_out["boosted_tau"]["post_trigger_num"]/trigger_out["boosted_tau"]["pre_trigger_num"]))
						if (trigger_bit == 39):
							eff_arr[0] = trigger_out["boosted_tau"]["post_trigger_num"]/trigger_out["boosted_tau"]["pre_trigger_num"]	
						if (trigger_bit == 40):
							eff_arr[1] = trigger_out["boosted_tau"]["post_trigger_num"]/trigger_out["boosted_tau"]["pre_trigger_num"]	
					else:
						#trigger_out[background_name]["boosted_tau"][var_name].plot1d(ax=ax)
						trigger_out[background_name][var_name].plot1d(ax=ax)
						print("Efficiency = %f"%(trigger_out[background_name]["post_trigger_num"]/trigger_out[background_name]["pre_trigger_num"]))
						if (trigger_bit == 39):
							eff_arr[0] = trigger_out[background_name]["post_trigger_num"]/trigger_out[background_name]["pre_trigger_num"]	
						if (trigger_bit == 40):
							eff_arr[1] = trigger_out[background_name]["post_trigger_num"]/trigger_out[background_name]["pre_trigger_num"]	
	
					if (hist_name_arr[0][-14:] == "NoTrigger_Plot"):
						plt.title(hist_name_arr[1] + title, wrap=True)
					else:
						plt.title(hist_name_arr[1] + " (" + trigger_name + r"), " + title, wrap=True)
					plt.savefig(hist_name_arr[0] + "-" + background_name + "-" + trigger_name)
					print(background_name)
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
		
			table_dict[title_dict[background_name]] = eff_arr


		#Set up table
		DrawTable(table_title,table_file_name,table_dict)

	

	


		
