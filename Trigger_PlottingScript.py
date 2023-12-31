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

def di_mass(part1,part2):
	return np.sqrt((part1.E + part2.E)**2 - (part1.Px + part2.Px)**2 - (part1.Py + part2.Py)**2 - (part1.Pz + part2.Pz)**2)

def bit_mask(in_bits):
	mask = 0
	for bit in in_bits:
		mask += (1 << bit)
	return mask

def dilep_di_mass(leptons):
	n_pair = np.floor(leptons.n/2)
	lep_minus = leptons(leptons.charge < 0)
	lep_plus = leptons(leptons.charge > 0)
	

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
				"iso": events.boostedTauByIsolationMVArun2v1DBoldDMwLTrawNew,
				"decay": events.boostedTaupfTausDiscriminationByDecayModeFinding,
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
				"eta": events.AK8JetEta,
				"phi": events.AK8JetPhi,
				"nEle": events.nEle,
				"trigger": events.HLTJet,
			},
			with_name="AK8JetArray",
			behavior=candidate.behavior,
		)
		
		Muon = ak.zip(
			{
				"n": events.nMu,
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
				"n": events.nEle,
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
				"PtTotUncDown": events.jetPtTotUncDown,
				"PtTotUncUp": events.jetPtTotUncUp,
				"PFLooseId": events.jetPFLooseId,
				#"HT": ak.sum(events.jetPt, axis=1),
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
		AK8Pt_NoCut = hist.Hist.new.Reg(40, 0, 1800, name = "JetPt_NoCut", label = r"AK8Jet $p_T$ [GeV]").Double()
		AK8Pt_Trigg = hist.Hist.new.Reg(40, 0, 1100, name = "JetPt_Trigg", label = r"AK8Jet $p_T$ [GeV]").Double()
		AK8SoftMass_PreTrigg = hist.Hist.new.Reg(40, 0, 300, name = "SoftMass_Trigg", label = "AK8Jet Soft Mass [GeV]").Double()
		AK8SoftMass_NoCut = hist.Hist.new.Reg(40, 0, 300, name = "SoftMass_NoCut", label = "AK8Jet Soft Mass [GeV]").Double()
		AK8SoftMass_Trigg = hist.Hist.new.Reg(40, 0, 300, name = "SoftMass_Trigg", label = "AK8Jet Soft Mass [GeV]").Double()		
		AK8JetMult_NoCut = hist.Hist.new.Reg(8,0,7, name = "AK8 Jet Multiplicity", label = "AK8 Jet Multiplicity").Double()
		AK8JetMult_PreTrigg = hist.Hist.new.Reg(8,0,7, name = "AK8 Jet Multiplicity", label = "AK8 Jet Multiplicity").Double()
		AK8JetMult_Trigg = hist.Hist.new.Reg(8,0,7, name = "AK8 Jet Multiplicity", label = "AK8 Jet Multiplicity").Double()
		
		#2D Histograms
		AK8Jet_PreTrigger = hist.Hist(
						hist.axis.Regular(20, 0, 1100, name="JetPt", label=r"AK8Jet $p_T$ [GeV]"),
						hist.axis.Regular(10, 0, 300, name="SoftMass", label=r"AK8Jet Soft Mass [GeV]")
					)		
		AK8Jet_Trigger = hist.Hist(
						hist.axis.Regular(20, 0, 1100, name="JetPt", label=r"AK8Jet $p_T$ [GeV]"),
						hist.axis.Regular(10, 0, 300, name="SoftMass", label="AK8Jet Soft Mass [GeV]")
					)		
		eff_AK8Jet = hist.Hist(
						hist.axis.Regular(20, 0, 1100, name="JetPt", label=r"AK8Jet $p_T$ [GeV]"),
						hist.axis.Regular(10, 0, 300, name="SoftMass", label="AK8Jet Soft Mass [GeV]")
					)		

		#Histograms (MET and HT) (Trigger bit = 39)
		HT_PreTrigg = hist.Hist.new.Reg(40, 0, 4000., label = "HT [GeV]").Double()
		HT_NoCut = hist.Hist.new.Reg(40, 0,4000., label = "HT [GeV]").Double()
		HT_NoCrossCleaning = hist.Hist.new.Reg(40, 0, 4000., label = "HT [GeV]").Double()
		HT_Trigg = hist.Hist.new.Reg(40, 0, 4000., label = "HT [GeV]").Double()
		MET_PreTrigg = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()
		MET_NoCut = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()
		MET_NoCrossCleaning = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()
		MET_Trigg = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()
		MHT_PreTrigg = hist.Hist.new.Reg(30, 0, 1200., name="MHT", label="MHT [GeV]").Double()
		MHT_NoCut = hist.Hist.new.Reg(30, 0, 1200., name="MHT", label="MHT [GeV]").Double()
		MHT_NoCrossCleaning = hist.Hist.new.Reg(30, 0, 1200., name="MHT", label="MHT [GeV]").Double()
		MHT_Trigg = hist.Hist.new.Reg(30, 0, 1200., name="MHT", label="MHT [GeV]").Double()

		#2D Histograms
		Jet_PreTrigger = hist.Hist(
			hist.axis.Regular(20, 0, 1200., name = "pfMET" , label = r"MET [GeV]"),
			hist.axis.Regular(20, 0, 4000., name = "HT", label = r"HT [GeV]")
		)
		Jet_Trigger = hist.Hist(
			hist.axis.Regular(20, 0, 1200., name = "pfMET" , label = r"MET [GeV]"),
			hist.axis.Regular(20, 0, 4000., name = "HT", label = r"HT [GeV]")
		)
		eff_Jet = hist.Hist(
			hist.axis.Regular(20, 0, 1200., name = "pfMET" , label = r"MET [GeV]"),
			hist.axis.Regular(20, 0, 4000., name = "HT", label = r"HT [GeV]")
		)

		
		#MHT
		Jet_MHT = Jet[Jet.Pt > 30]
		Jet_MHT = Jet[np.abs(Jet.eta) < 5]
		Jet_MHT = Jet_MHT[Jet_MHT.PFLooseId > 0.5]
		#JetUp_MHT = Jet[Jet.PtTotUncUp > 30]
		#JetUp_MHT = JetUp_MHT[np.abs(JetUp_MHT.eta) < 5]
		#JetUp_MHT = JetUp_MHT[JetUp_MHT.PFLooseId > 0.5]
		#JetDown_MHT = Jet[Jet.PtTotUncDown > 30]
		#JetDown_MHT = JetDown_MHT[np.abs(JetDown_MHT.eta) < 5]
		#JetDown_MHT = JetDown_MHT[JetDown_MHT.PFLooseId > 0.5]
		Jet_MHT["MHT_x"] = ak.sum(Jet_MHT.Pt*np.cos(Jet_MHT.phi),axis=1,keepdims=False) #+ ak.sum(JetUp_MHT.PtTotUncUp*np.cos(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.cos(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet_MHT["MHT_y"] = ak.sum(Jet_MHT.Pt*np.sin(Jet_MHT.phi),axis=1,keepdims=False) #+ ak.sum(JetUp_MHT.PtTotUncUp*np.sin(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.sin(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet_MHT["MHT"] = np.sqrt(Jet_MHT.MHT_x**2 + Jet_MHT.MHT_y**2)
		MHT_NoCrossCleaning.fill(ak.ravel(Jet_MHT.MHT))
		MHT_Acc_NoCrossingClean = hist.accumulators.Mean().fill(ak.ravel(Jet_MHT.MHT))
		print("Jet MHT Defined:")
		
		#HT Seleciton (new)
		#tau_temp1,HT = ak.unzip(ak.cartesian([tau,Jet_MHT], axis = 1, nested = True))
		Jet_HT = Jet[Jet.Pt > 30]
		Jet_HT = Jet_HT[np.abs(Jet_HT.eta) < 3]
		Jet_HT = Jet_HT[Jet_HT.PFLooseId > 0.5]
		#JetUp_HT = Jet[Jet.PtTotUncUp > 30]
		#JetUp_HT = JetUp_HT[np.abs(JetUp_HT.eta) < 3]
		#JetUp_HT = JetUp_HT[JetUp_HT.PFLooseId > 0.5]
		#JetDown_HT = Jet[Jet.PtTotUncDown > 30]
		#JetUp_HT = JetDown_HT[np.abs(JetDown_HT.eta) < 3]
		#JetDown_HT = JetDown_HT[JetDown_HT.PFLooseId > 0.5]
		HT,tau_temp1 = ak.unzip(ak.cartesian([Jet_HT,tau], axis = 1, nested = True))
		#HT_up,tau_temp2 = ak.unzip(ak.cartesian([JetUp_HT,tau], axis = 1, nested = True))
		#HT_down,tau_temp3 = ak.unzip(ak.cartesian([JetDown_HT,tau], axis = 1, nested = True))
		
		#Get Cross clean free histograms
		HT_Var_NoCrossClean = ak.sum(Jet_HT.Pt,axis = 1,keepdims=True) #+ ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=True) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=True)
		MET_NoCrossCleaning.fill(ak.ravel(Jet.pfMET))
		MET_Acc_NoCrossClean = hist.accumulators.Mean().fill(ak.ravel(Jet.pfMET))
		#HT_NoCrossCleaning.fill(ak.sum(Jet_MHT.Pt,axis = 1,keepdims=False) + ak.sum(JetUp_MHT.PtTotUncUp,axis = 1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown,axis = 1,keepdims=False))	
		#HT_NoCrossCleaning.fill(ak.ravel(HT_Var_NoCrossClean[HT_Var_NoCrossClean > 0]))
		HT_NoCrossCleaning.fill(ak.ravel(HT_Var_NoCrossClean))
		HT_num = 0
		for x in ak.sum(Jet.Pt,axis = 1,keepdims=False):
			HT_num += 1
			
		print(ak.sum(Jet.Pt,axis = 1,keepdims=True))
		print("HT Num (No Cross Cleaning): %d"%HT_num)
		#HT_Acc_NoCrossClean = hist.accumulators.Mean().fill(ak.ravel(HT_Var_NoCrossClean[HT_Var_NoCrossClean > 0]))
		HT_Acc_NoCrossClean = hist.accumulators.Mean().fill(ak.ravel(HT_Var_NoCrossClean))
	
		mval_temp = deltaR(tau_temp1,HT) >= 0.5
		print(mval_temp)
		if (len(Jet.Pt) != len(mval_temp)):
			print("Things aren't good")
			if (len(Jet.Pt) > len(mval_temp)):
				print("More Jets than entries in mval_temp")
			if (len(Jet.Pt) < len(mval_temp)):
				print("Fewer entries in Jets than mval_temp")

		Jet_HT["dR"] = mval_temp
		HT_Val_NoCuts = ak.sum(Jet_HT.Pt,axis = 1,keepdims=True) #+ ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=True) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=True)
		Jet["HT"] = ak.sum(Jet_HT.Pt,axis = 1,keepdims=False) #+ ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=False) + ak.sum(JetDown_HT.PtTotUncDown,axis = 1,keepdims=False)

		test_HT = ak.sum(Jet.Pt,axis = 1,keepdims=True)
		HT_num = 0
		print("Test 1:")
		print(ak.sum(Jet.Pt,axis = 1,keepdims=False))
		print("Test 2:")
		print(ak.sum(Jet.Pt,axis = 1,keepdims=True))
		print(ak.ravel(Jet.HT))
		print(Jet.HT)

		#Histograms of variables relavent to trigger 
		if (self.trigger_bit == 40):
			AK8Jet_Temp = AK8Jet[ak.num(AK8Jet, axis = 1) > 0]
			AK8Pt_NoCut.fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetPt)) #Change to leading pt
			AK8Pt_Acc = hist.accumulators.Mean().fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetPt))
			AK8SoftMass_NoCut.fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetDropMass))
			AK8SoftMass_Acc = hist.accumulators.Mean().fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetDropMass))
			AK8JetMult_NoCut.fill(ak.num(AK8Jet, axis=1))
			
		if (self.trigger_bit == 39):
			print("No Cut Accumulators updated")
			#HT_NoCut.fill(ak.ravel(HT_Val_NoCuts[HT_Val_NoCuts > 0]))
			#HT_Acc_NoCut = hist.accumulators.Mean().fill(ak.ravel(HT_Val_NoCuts[HT_Val_NoCuts > 0]))
			HT_NoCut.fill(ak.ravel(HT_Val_NoCuts))
			HT_Acc_NoCut = hist.accumulators.Mean().fill(ak.ravel(HT_Val_NoCuts))
			print("HT Mean: %f"%HT_Acc_NoCut.value)
			MET_NoCut.fill(ak.ravel(Jet.pfMET))
			MET_Acc_NoCut = hist.accumulators.Mean().fill(ak.ravel(Jet.pfMET))
			print("MET Mean: %f"%MET_Acc_NoCut.value)
				

		trigger_mask = bit_mask([self.trigger_bit])		
		tau = tau[tau.pt > 30] #pT
		tau = tau[np.abs(tau.eta) < 2.3] #eta
		
		#Loose isolation
		tau = tau[tau.decay >= 0.5]	
		tau = tau[tau.iso >= 0.5]


		#Delta R Cut on taus (Based on metric table https://github.com/CoffeaTeam/coffea/blob/f7e9119ba4567ba6a5e593da77627af475eae8e9/coffea/nanoevents/methods/vector.py#L668)
		a,b = ak.unzip(ak.cartesian([tau,tau], axis = 1, nested = True))
		mval = deltaR(a,b) < 0.8 
		tau["dRCut"] = mval
		tau = tau[ak.any(tau.dRCut, axis = 2) == True]	
		
		AK8Jet = AK8Jet[(ak.sum(tau.charge,axis=1) == 0)] #Apply charge conservation cut to AK8Jets
		Muon = Muon[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		Electron = Electron[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		Jet = Jet[(ak.sum(tau.charge,axis=1) == 0)]
		Jet_HT = Jet_HT[(ak.sum(tau.charge,axis=1) == 0)]
		#JetUp_HT = JetUp_HT[(ak.sum(tau.charge,axis=1) == 0)]
		#JetDown_HT = JetDown_HT[(ak.sum(tau.charge,axis=1) == 0)]
		Jet_MHT = Jet_MHT[(ak.sum(tau.charge,axis=1) == 0)]
		tau = tau[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		
		#Investegate entries with more than 4 taus
		# n_more = len(tau[ak.num(tau) > 4])
		# print("Events with more than 4 taus: %d"%n_more)
		
		# if (n_more > 0):
		# 	print("========!!Important information about events with more than 4 tau!!========")
		# 	diff_num = n_more
		# 	test_obj = tau[ak.num(tau) > 4] 
		# 	n = 5
		# 	while(n_more > 0):
		# 		N_events = len(test_obj[ak.num(test_obj) == 5])
		# 		print("Number of events with %d taus: %d"%(n,N_events))
		# 		n +=1
		# 		diff_num -= N_events
		

		AK8Jet = AK8Jet[ak.num(tau) >= 4]
		Electron = Electron[ak.num(tau) >= 4]
		Muon = Muon[ak.num(tau) >= 4]
		Jet_MHT = Jet_MHT[ak.num(tau) >= 4]
		Jet_HT = Jet_HT[ak.num(tau) >= 4]
		#JetUp_HT = JetUp_HT[ak.num(tau) >= 4]
		#JetDown_HT = JetDown_HT[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		tau = tau[ak.num(tau) >= 4] #4 tau events
		

		if (self.trigger_bit == 40):
			AK8Pt_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetPt))
			AK8Pt_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetPt)
			AK8SoftMass_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetDropMass))
			AK8SoftMass_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetDropMass)
			AK8Pt_all.fill("No Trigger",ak.ravel(AK8Jet.AK8JetPt))
			AK8JetMult_PreTrigg.fill(ak.num(AK8Jet,axis=1))
		
		if (self.trigger_bit == 39):
			#Fill Histograms
			HT_Val_PreTrigger = ak.sum(Jet_HT.Pt, axis = 1, keepdims=True) #+ ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=True) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=True)
			Jet["HT"] = ak.sum(Jet_HT.Pt, axis = 1, keepdims=False) #+ ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=False) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=False)
			HT_PreTrigg.fill(ak.ravel(HT_Val_PreTrigger[HT_Val_PreTrigger > 0]))
			HT_PreTrigg.fill(ak.ravel(HT_Val_PreTrigger))
			HT_NoTrigg_Arr = ak.ravel(Jet.HT)
			MET_PreTrigg.fill(ak.ravel(Jet.pfMET))
			MHT_PreTrigg.fill(ak.ravel(Jet_MHT.MHT))
			MET_NoTrigg_Arr = ak.ravel(Jet.pfMET)
			print("MET Len: %d"%len(MET_NoTrigg_Arr))	
			print("HT Len: %d"%len(HT_NoTrigg_Arr))	

		#Apply Online trigger
		tau = tau[np.bitwise_and(tau.trigger,trigger_mask) == trigger_mask]
		AK8Jet = AK8Jet[np.bitwise_and(AK8Jet.trigger,trigger_mask) == trigger_mask]
		Jet = Jet[np.bitwise_and(Jet.trigger,trigger_mask) == trigger_mask]
		Jet_HT = Jet_HT[np.bitwise_and(Jet_HT.trigger,trigger_mask) == trigger_mask]
		Jet_MHT = Jet_MHT[np.bitwise_and(Jet_MHT.trigger,trigger_mask) == trigger_mask]
		tau_plus = tau[tau.charge > 0]	
		tau_minus = tau[tau.charge < 0]

		#Construct all possible valid ditau pairs
		tau_plus1, tau_plus2 = ak.unzip(ak.combinations(tau_plus,2))
		tau_minus1, tau_minus2 = ak.unzip(ak.combinations(tau_minus,2))
		
		deltaR11 = deltaR(tau_plus1, tau_minus1)
		deltaR12 = deltaR(tau_plus1, tau_minus2)
		deltaR22 = deltaR(tau_plus2, tau_minus2)
		deltaR21 = deltaR(tau_plus2, tau_minus1)
		
		#Efficiency Histograms 
		if (self.trigger_bit == 40):	
			AK8Pt_Trigg.fill(ak.ravel(AK8Jet.AK8JetPt))
			AK8SoftMass_Trigg.fill(ak.ravel(AK8Jet.AK8JetDropMass))
			AK8Pt_all.fill("Trigger",ak.ravel(AK8Jet.AK8JetPt))	
			AK8Pt_Trigg_Arr = ak.ravel(AK8Jet.AK8JetPt)
			AK8SoftMass_all.fill("Trigger",ak.ravel(AK8Jet.AK8JetDropMass))	
			AK8SoftMass_Trigg_Arr = ak.ravel(AK8Jet.AK8JetDropMass)
			pre_triggernum = ak.num(AK8Pt_NoTrigg_Arr,axis=0)
			post_triggernum = ak.num(AK8Pt_Trigg_Arr,axis=0)
			AK8JetMult_Trigg.fill(ak.num(AK8Jet,axis=1))
	
			if (self.signal):
				print("Efficiency (AK8Jet Trigger): %f"%(ak.num(AK8Pt_Trigg_Arr,axis=0)/ak.num(AK8Pt_NoTrigg_Arr,axis=0)))
			AK8Jet_PreTrigger.fill(AK8Pt_NoTrigg_Arr, AK8SoftMass_NoTrigg_Arr)
			AK8Jet_Trigger.fill(AK8Pt_Trigg_Arr, AK8SoftMass_Trigg_Arr)
			eff_AK8Jet = AK8Jet_Trigger/AK8Jet_PreTrigger
		
		if (self.trigger_bit == 39):
			HT_Val_PostTrigger = ak.sum(Jet_HT.Pt, axis = 1, keepdims=True)
			HT_Trigg.fill(ak.ravel(HT_Val_PostTrigger[HT_Val_PostTrigger > 0]))
			HT_Trigg_Arr = ak.ravel(HT_Val_PostTrigger[HT_Val_PostTrigger > 0])
			#HT_Trigg.fill(ak.ravel(HT_Val_PostTrigger))
			#HT_Trigg_Arr = ak.ravel(HT_Val_PostTrigger)
			HT_Trigg_Arr = ak.ravel(Jet.HT)
			MET_Trigg.fill(ak.ravel(Jet.pfMET))
			MHT_Trigg.fill(ak.ravel(Jet_MHT.MHT))
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
					"AK8JetPt_NoCut": AK8Pt_NoCut, 
					"AK8JetSoftMass_NoCut": AK8SoftMass_NoCut,
					"Acc_AK8JetPt_NoCut": AK8Pt_Acc,
					"Acc_AK8JetSoftMass_NoCut": AK8SoftMass_Acc,
					"AK8JetMult_NoCut": AK8JetMult_NoCut,
					"AK8JetMult_PreTrigg": AK8JetMult_PreTrigg,
					"AK8JetMult_Trigg": AK8JetMult_Trigg
				}
			}
		if (self.trigger_bit == 39):
			return{
				 dataset: {
					"MET_PreTrigg": MET_PreTrigg,
					"MHT_PreTrigg": MHT_PreTrigg,
					"MET_Trigg": MET_Trigg,
					"MHT_Trigg": MHT_Trigg,
					"HT_PreTrigg": HT_PreTrigg,
					"HT_Trigg": HT_Trigg,
					"Jet_PreTrigg": Jet_PreTrigger,
					"Jet_Trigg": Jet_Trigger,
					"Jet_eff": eff_Jet,
					"pre_trigger_num": pre_triggernum,
					"post_trigger_num": post_triggernum,
					"MET_NoCut": MET_NoCut,
					"Acc_MET_NoCut": MET_Acc_NoCut,
					"HT_NoCut": HT_NoCut,
					"Acc_HT_NoCut": HT_Acc_NoCut,
					"MET_NoCrossClean": MET_NoCrossCleaning,
					"MHT_NoCrossClean": MHT_NoCrossCleaning,
					"Acc_MET_NoCrossClean": MET_Acc_NoCrossClean,
					"Acc_MHT_NoCrossClean": MHT_Acc_NoCrossingClean,
					"HT_NoCrossClean": HT_NoCrossCleaning,
					"Acc_HT_NoCrossClean": HT_Acc_NoCrossClean
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
				"iso": events.boostedTauByIsolationMVArun2v1DBoldDMwLTrawNew,
				"decay": events.boostedTaupfTausDiscriminationByDecayModeFinding,
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
            .Double()
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
		tau = tau[np.abs(tau.eta) < 2.3] #eta
		
		#Loose isolation
		tau = tau[tau.decay >= 0.5]		
		tau = tau[tau.iso >= 0.5]
		
		#Delta R Cut on taus
		a,b = ak.unzip(ak.cartesian([tau,tau], axis = 1, nested = True))
		mval = deltaR(a,b) < 0.8 
		tau["dRCut"] = mval
		tau = tau[ak.any(tau.dRCut, axis = 2) == True]	
		
		tau = tau[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		
		#print("Before 4 tau cut length is: %d" % len(tau))
		#Investegate entries with more than 4 taus
		# n_more = len(tau[ak.num(tau) > 4])
		# print("Events with more than 4 taus: %d"%n_more)
		
		# if (n_more > 0):
		# 	print("========!!Important information about events with more than 4 tau!!========")
		# 	diff_num = n_more
		# 	test_obj = tau[ak.num(tau) > 4] 
		# 	n = 5
		# 	while(n_more > 0):
		# 		N_events = len(test_obj[ak.num(test_obj) == 5])
		# 		print("Number of events with %d taus: %d"%(n,N_events))
		# 		n +=1
		# 		diff_num -= N_events
		
		tau = tau[ak.num(tau) >= 4] #4 tau events (unsure about this)	
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

		#print(deltaR11 < 0.8)
		#print((deltaR11 < deltaR12) and (deltaR11 < deltaR21) and deltaR11 < 0.8)
		print("Delta R11 length = %d"%len(deltaR11))
		print("tau array length = %d"%len(tau))

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
		pt_leading_acc = hist.accumulators.Mean().fill(leading_tau.pt)
		pt_all_hist.fill("Subleading",subleading_tau.pt)
		pt_subleading_acc = hist.accumulators.Mean().fill(subleading_tau.pt)
		pt_all_hist.fill("Third-leading",thirdleading_tau.pt)
		pt_thirdleading_acc = hist.accumulators.Mean().fill(thirdleading_tau.pt)
		pt_all_hist.fill("Fourth-leading",fourthleading_tau.pt)
		pt_fourthleading_acc = hist.accumulators.Mean().fill(fourthleading_tau.pt)
		pt_all_hist *= (1/pt_all_hist.sum())
		pt4_hist.fill(fourthleading_tau.pt)
		
		#Ditau mass plots 
		dimass_all_hist.fill("Leading pair", ak.ravel(di_mass(tau_plus1[(deltaR11 < deltaR21)], tau_minus1[(deltaR11 < deltaR21)])))
		dimass_all_hist.fill("Leading pair", ak.ravel(di_mass(tau_plus2[(deltaR21 < deltaR11)], tau_minus1[(deltaR21 < deltaR11)])))
		dimass_leading_acc = hist.accumulators.Mean().fill(ak.concatenate([ak.ravel(di_mass(tau_plus1[(deltaR11 < deltaR21)], tau_minus1[(deltaR11 < deltaR21)])), ak.ravel(di_mass(tau_plus2[(deltaR21 < deltaR11)], tau_minus1[(deltaR21 < deltaR11)]))]))
		dimass_all_hist.fill("Subleading pair", ak.ravel(di_mass(tau_plus1[(deltaR12 < deltaR22)], tau_minus2[(deltaR12 < deltaR22)])))
		dimass_all_hist.fill("Subleading pair", ak.ravel(di_mass(tau_plus2[(deltaR22 < deltaR12)], tau_minus2[(deltaR22 < deltaR12)])))
		dimass_subleading_acc = hist.accumulators.Mean().fill(ak.concatenate([ak.ravel(di_mass(tau_plus1[(deltaR12 < deltaR22)], tau_minus2[(deltaR12 < deltaR22)])), ak.ravel(di_mass(tau_plus2[(deltaR22 < deltaR12)], tau_minus2[(deltaR22 < deltaR12)]))]))
		dimass_all_hist *= 1/(dimass_all_hist.sum())		
	
		ditau_mass1_hist.fill(ak.ravel(di_mass(tau_plus1[(deltaR11 < deltaR21)], tau_minus1[(deltaR11 < deltaR21)])))	
		ditau_mass1_hist.fill(ak.ravel(di_mass(tau_plus1[(deltaR21 < deltaR11)], tau_minus2[(deltaR21 < deltaR11)])))
		ditau_mass1_hist *= (1/ditau_mass1_hist.sum())
		ditau_mass2_hist.fill(ak.ravel(di_mass(tau_plus2[(deltaR22 < deltaR12)], tau_minus2[(deltaR22 < deltaR12)])))	
		ditau_mass2_hist.fill(ak.ravel(di_mass(tau_plus2[(deltaR12 < deltaR22)], tau_minus1[(deltaR12 < deltaR22)])))
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
				"ditau_mass": dimass_all_hist,
				"pt_leading_acc": pt_leading_acc,
				"pt_subleading_acc": pt_subleading_acc,
				"pt_thirdleading_acc": pt_thirdleading_acc,
				"pt_fourthleading_acc": pt_fourthleading_acc,
				"dimass_leading_acc": dimass_leading_acc,
				"dimass_subleading_acc": dimass_subleading_acc
				
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
		"AK8JetPt_Trigg" : ["AK8Pt_Trigger_Plot",r"AK8Jet $p_T$ Trigger"], "AK8JetPt_PreTrigg" : ["AK8Pt_NoTrigger_Plot",r"AK8Jet $p_T$ No Trigger"],
		"AK8JetPt_NoCut" : ["AK8Pt_NoCut_Plot", r"AK8Jet $p_T$ No Cuts"], "AK8JetSoftMass_NoCut" : ["AK8SoftMass_NoCut_Plot", "AK8SoftDrop Mass No Cut"],
		"AK8JetMult_PreTrigg" : ["AK8JetMult_NoTrigger_Plot", "AK8Jet Multiplicity"], "AK8JetMult_Trigg" : ["AK8JetMult_Trigger_Plot", "AK8Jet Multiplicity"],
		"AK8JetMult_NoCut" : ["AK8JetMult_NoCut_Plot", "AK8Jet Multiplicity"]
	}
	
	trigger_AK8Jet_hist_dict_2d = {
		"AK8Jet_PreTrigg" : ["AK8Jet_PreTriggerHist_Plot", "AK8Jet 2D Histogram No Trigger"], "AK8Jet_Trigg" : ["AK8Jet_TriggerHist_Plot", "AK8Jet 2D Histogram Trigger"],
		"AK8Jet_eff" : ["AK8Jet_Eff_Plot", "AK8Jet 2D Efficiency Histogram Trigger"]
	}

	trigger_MTHTJet_hist_dict_1d = {
		"MET_Trigg" : ["MET_Trigger_Plot","pfMET Trigger"] , "MET_PreTrigg" : ["MET_NoTrigger_Plot","pfMET No Trigger"], "MET_NoCut": ["MET_NoCut_Plot", "pfMET No Cuts/Selections"], 
		"MHT_Trigg" : ["MHT_Trigger_Plot","MHT Trigger"] , "MHT_PreTrigg" : ["MHT_NoTrigger_Plot","MHT No Trigger"],
		"HT_Trigg" : ["HT_Trigger_Plot",r"HT Trigger"], "HT_PreTrigg" : ["HT_NoTrigger_Plot", r"HT No Trigger"], "HT_NoCut" : ["HT_NoCut_Plot", "HT No Cuts/Selections"],
		"MET_NoCrossClean" : ["MET_NoCrossClean_Plot", "pfMET No Cross Cleaning Applied"], "MHT_NoCrossClean" : ["MHT_NoCrossClean_Plot", "MHT No Cross Cleaning Applied"], "HT_NoCrossClean" : ["HT_NoCrossComp_Plot", "HT No Cross Cleaing Applied"]
	}
	
	trigger_MTHTJet_hist_dict_2d = {
		"Jet_PreTrigg" : ["Jet_PreTriggerHist_Plot", "MET and HT 2D Histogram No Trigger"], "Jet_Trigg" : ["Jet_TriggerHist_Plot", "MET and HT 2D Histogram Trigger"],
		"Jet_eff" : ["Jet_Eff_Plot", "MET and HT Efficiency Histogram Trigger"]
	}
	
	trigger_dict = {"PFHT500_PFMET100_PFMHT100_IDTight": 39, "AK8PFJet400_TrimMass30": 40}

	signal_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v1_Hadd/GluGluToRadionToHHTo4T_M-"
	
	#Signal
	for mass_str in mass_str_arr:
		fileName = signal_base + mass_str + ".root"
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
				plt.text(x = 0.5,y = 0.7, s = r"Mean leading Di-$\tau$ mass: %.2f GeV"%out["boosted_tau"]["dimass_leading_acc"].value, transform = ax.transAxes, fontsize="small")
				plt.text(x = 0.5,y = 0.65, s = r"Mean subleading Di-$\tau$ mass: %.2f GeV"%out["boosted_tau"]["dimass_subleading_acc"].value, transform = ax.transAxes, fontsize="small")
			if (hist_name_arr[0] == "AllPt_Plot"):
				ax.legend(title=r"$\tau$")
				plt.text(x = 0.55, y = 0.65, s = r"Mean leading $\tau$ $p_T$: %.2f GeV"%out["boosted_tau"]["pt_leading_acc"].value, transform = ax.transAxes, fontsize="small")
				plt.text(x = 0.55, y = 0.6, s = r"Mean subleading $\tau$ $p_T$: %.2f GeV"%out["boosted_tau"]["pt_subleading_acc"].value, transform = ax.transAxes, fontsize="small")
				plt.text(x = 0.55, y = 0.55, s = r"Mean third-leading $\tau$ $p_T$: %.2f GeV"%out["boosted_tau"]["pt_thirdleading_acc"].value, transform = ax.transAxes, fontsize="small")
				plt.text(x = 0.55, y = 0.5, s = r"Mean fourth-leading $\tau$ $p_T$: %.2f GeV"%out["boosted_tau"]["pt_fourthleading_acc"].value, transform = ax.transAxes, fontsize="small")
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
			
				#Add Text with average and number of entries	
				if (var_name == "MET_NoCut"):
					plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out["boosted_tau"]["Acc_MET_NoCut"].count, transform = ax.transAxes)
					plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out["boosted_tau"]["Acc_MET_NoCut"].value, transform = ax.transAxes)
				if (var_name == "MET_NoCrossClean"):
					plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out["boosted_tau"]["Acc_MET_NoCrossClean"].count, transform = ax.transAxes)
					plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out["boosted_tau"]["Acc_MET_NoCrossClean"].value, transform = ax.transAxes)
				if (var_name == "HT_NoCut"):
					plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out["boosted_tau"]["Acc_HT_NoCut"].count, transform = ax.transAxes)
					plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out["boosted_tau"]["Acc_HT_NoCut"].value, transform = ax.transAxes)
				if (var_name == "HT_NoCrossClean"):
					plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out["boosted_tau"]["Acc_HT_NoCrossClean"].count, transform = ax.transAxes)
					plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out["boosted_tau"]["Acc_HT_NoCrossClean"].value, transform = ax.transAxes)
				if (var_name == "AK8JetPt_NoCut"):
					plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out["boosted_tau"]["Acc_AK8JetPt_NoCut"].count, transform = ax.transAxes)
					plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out["boosted_tau"]["Acc_AK8JetPt_NoCut"].value, transform = ax.transAxes)
				if (var_name == "AK8JetSoftMass_NoCut"):
					plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out["boosted_tau"]["Acc_AK8JetSoftMass_NoCut"].count, transform = ax.transAxes)
					plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out["boosted_tau"]["Acc_AK8JetSoftMass_NoCut"].value, transform = ax.transAxes)
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
	background_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/"
	#background_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedH3/2018/tt/v2_fast_Hadd/"
	background_dict = {"ZZ4l" : r"$ZZ \rightarrow 4l$", "top": "Top Background"}
	#file_dict = {"ZZ4l": [file_base + "ZZ4l.root"], "top": [file_base + "Tbar-tchan.root",file_base + "Tbar-tW.root",file_base + "T-tchan.root"]}
	#file_dict = {"top": [file_base + "Tbar-tchan.root",file_base + "Tbar-tW.root",file_base + "T-tchan.root"]}
	file_dict = {"top": [background_base + "TTTo2L2Nu.root",background_base + "TTToSemiLeptonic.root",background_base + "TTToHadronic.root"]}
	#file_dict = {"top": [background_base + "TTTo2L2Nu.root"]} #Single top file

	iterative_runner = processor.Runner(
		executor = processor.IterativeExecutor(compression=None),
		schema=BaseSchema
	)

	for background_name, title in background_dict.items():
		if (background_name == "ZZ4l"):
			events = NanoEventsFactory.from_root(
				background_base + background_name + ".root",
				treepath="/4tau_tree",
				#treepath="/tautau_tree",
				schemaclass = BaseSchema,
				metadata={"dataset": "boosted_tau"},
			).events()
		
		print("Background: " + background_name)	
		for trigger_name, trigger_bit in trigger_dict.items():
			if (background_name == "ZZ4l"):
				p2 = TriggerStudies(trigger_bit, False)
				trigger_out = p2.process(events)
				out_name = "boosted_tau"
			else:
				trigger_out = iterative_runner(file_dict, treename="4tau_tree",processor_instance=TriggerStudies(trigger_bit, False)) 
				out_name = background_name
				#trigger_out = iterative_runner(file_dict, treename="tautau_tree",processor_instance=TriggerStudies(trigger_bit, False)) 
			
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
				else:
					#trigger_out[background_name]["boosted_tau"][var_name].plot1d(ax=ax)
					trigger_out[background_name][var_name].plot1d(ax=ax)
					print("Efficiency = %f"%(trigger_out[background_name]["post_trigger_num"]/trigger_out[background_name]["pre_trigger_num"]))
	
				if (hist_name_arr[0][-14:] == "NoTrigger_Plot"):
					plt.title(hist_name_arr[1] + title, wrap=True)
				else:
					plt.title(hist_name_arr[1] + " (" + trigger_name + r"), " + title, wrap=True)
				
				#Add Text with average and number of entries	
				if (var_name == "MET_NoCut"):
					plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out[out_name]["Acc_MET_NoCut"].count, transform = ax.transAxes)
					plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out[out_name]["Acc_MET_NoCut"].value, transform = ax.transAxes)
				if (var_name == "MET_NoCrossClean"):
					plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out[out_name]["Acc_MET_NoCrossClean"].count, transform = ax.transAxes)
					plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out[out_name]["Acc_MET_NoCrossClean"].value, transform = ax.transAxes)
				if (var_name == "HT_NoCut"):
					plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out[out_name]["Acc_HT_NoCut"].count, transform = ax.transAxes)
					plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out[out_name]["Acc_HT_NoCut"].value, transform = ax.transAxes)
				if (var_name == "HT_NoCrossClean"):
					plt.text(x = 0.14, y = 0.8, s = "Entries: %d"%trigger_out[out_name]["Acc_HT_NoCrossClean"].count, transform = ax.transAxes)
					plt.text(x = 0.14, y = 0.74, s = "Mean: %.2f GeV"%trigger_out[out_name]["Acc_HT_NoCrossClean"].value, transform = ax.transAxes)
				if (var_name == "AK8JetPt_NoCut"):
					plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out[out_name]["Acc_AK8JetPt_NoCut"].count, transform = ax.transAxes)
					plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out[out_name]["Acc_AK8JetPt_NoCut"].value, transform = ax.transAxes)
				if (var_name == "AK8JetSoftMass_NoCut"):
					plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out[out_name]["Acc_AK8JetSoftMass_NoCut"].count, transform = ax.transAxes)
					plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out[out_name]["Acc_AK8JetSoftMass_NoCut"].value, transform = ax.transAxes)
				
				plt.savefig(hist_name_arr[0] + "-" + background_name + "-" + trigger_name)
				print(background_name)
				plt.close()
				  
			for var_name, hist_name_arr in trigger_hist_dict_2d.items():
				fig, ax = plt.subplots()
				if (background_name == "ZZ4l"):
					trigger_out["boosted_tau"][var_name].plot2d(ax=ax)
				else:
					#trigger_out[background_name]["boosted_tau"][var_name].plot2d(ax=ax)
					if ("eff" not in var_name):
						trigger_out[background_name][var_name].plot2d(ax=ax)
					if (var_name == "Jet_eff"):
						#Set up efficiency histogram
						eff_Jet = hist.Hist(
							hist.axis.Regular(20, 0, 1200., name = "pfMET" , label = r"MET [GeV]"),
							hist.axis.Regular(20, 0, 3500., name = "HT", label = r"HT [GeV]")
						)
						eff_Jet = trigger_out[background_name]["Jet_Trigg"]/trigger_out[background_name]["Jet_PreTrigg"] 
						eff_Jet.plot2d(ax=ax)
					if (var_name == "AK8Jet_eff"):
						print("AK8Jet Stuff!!")
						eff_AK8Jet = hist.Hist(
							hist.axis.Regular(20, 0, 1100, name="JetPt", label=r"AK8Jet $p_T$ [GeV]"),
							hist.axis.Regular(10, 0, 300, name="SoftMass", label="AK8Jet Soft Mass [GeV]")
						)		
						eff_AK8Jet = trigger_out[background_name]["AK8Jet_Trigg"]/trigger_out[background_name]["AK8Jet_PreTrigg"]		
						eff_AK8Jet.plot2d(ax=ax)
	
				if (hist_name_arr[0][-19:] == "PreTriggerHist_Plot"):
					plt.title(hist_name_arr[1] + title, wrap=True)
				else:
					plt.title(hist_name_arr[1] + trigger_name + "), " + title, wrap=True)
				plt.savefig(hist_name_arr[0] + "-" + background_name + "-" + trigger_name)
				plt.close()
		
