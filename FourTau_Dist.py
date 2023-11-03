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

def four_mass(part_list): #Four Particle mass assuming each event has 4 particles
	return np.sqrt(sum(part_list.E)**2 - sum(part_list.Px)**2 - sum(part_list.Py)**2 - sum(part_list.Pz)**2)

def bit_mask(in_bits):
	mask = 0
	for bit in in_bits:
		mask += (1 << bit)
	return mask

def bit_or(data):
	cond_1 = np.bitwise_and(data.trigger,bit_mask([39,40])) == bit_mask([39,40])
	cond_2 = np.bitwise_and(data.trigger,bit_mask([39,40])) == bit_mask([39])
	cond_3 = np.bitwise_and(data.trigger,bit_mask([39,40])) == bit_mask([40])
	return np.bitwise_or(cond_1, np.bitwise_or(cond_2,cond_3))

class FourTauPlotting(processor.ProcessorABC):
	def __init__(self, trigger_bit, trigger_cut = True, offline_cut = False, OrTrigger = False):
		self.trigger_bit = trigger_bit
		#self.signal = signal (This is useless now)
		self.offline_cut = offline_cut
		self.trigger_cut = trigger_cut
		self.OrTrigger = OrTrigger
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
		
		Jet = ak.zip(
			{
				"Pt": events.jetPt,
				"pfMET": events.pfMET,
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
		AK8Pt_NoCut = hist.Hist.new.Reg(40, 0, 1800, name = "JetPt_NoCut", label = r"AK8Jet $p_T$ [GeV]").Double()
		AK8Pt_Trigg = hist.Hist.new.Reg(40, 0, 1100, name = "JetPt_Trigg", label = r"AK8Jet $p_T$ [GeV]").Double()
		AK8SoftMass_PreTrigg = hist.Hist.new.Reg(40, 0, 300, name = "SoftMass_Trigg", label = "AK8Jet Soft Mass [GeV]").Double()
		AK8SoftMass_NoCut = hist.Hist.new.Reg(40, 0, 300, name = "SoftMass_NoCut", label = "AK8Jet Soft Mass [GeV]").Double()
		AK8SoftMass_Trigg = hist.Hist.new.Reg(40, 0, 300, name = "SoftMass_Trigg", label = "AK8Jet Soft Mass [GeV]").Double()		
		
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
		HT_PreTrigg = hist.Hist.new.Reg(40, 0, 3500., label = "HT [GeV]").Double()
		HT_NoCut = hist.Hist.new.Reg(40, 0, 3500., label = "HT [GeV]").Double()
		HT_NoCrossCleaning = hist.Hist.new.Reg(40, 0, 3500., label = "HT [GeV]").Double()
		HT_Trigg = hist.Hist.new.Reg(40, 0, 3500., label = "HT [GeV]").Double()
		MET_PreTrigg = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()
		MET_NoCut = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()
		MET_NoCrossCleaning = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()
		MET_Trigg = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()

		#Histograms 4 tau
		FourTau_Mass_hist = hist.Hist.new.Reg(40,10,300, label = r"$m_{4\tau} [GeV]$").Double()

		#2D Histograms
		Jet_PreTrigger = hist.Hist(
			hist.axis.Regular(20, 0, 1200., name = "pfMET" , label = r"MET [GeV]"),
			hist.axis.Regular(20, 0, 3500., name = "HT", label = r"HT [GeV]")
		)
		Jet_Trigger = hist.Hist(
			hist.axis.Regular(20, 0, 1200., name = "pfMET" , label = r"MET [GeV]"),
			hist.axis.Regular(20, 0, 3500., name = "HT", label = r"HT [GeV]")
		)
		eff_Jet = hist.Hist(
			hist.axis.Regular(20, 0, 1200., name = "pfMET" , label = r"MET [GeV]"),
			hist.axis.Regular(20, 0, 3500., name = "HT", label = r"HT [GeV]")
		)

		
		#MHT
		Jet_MHT = Jet[Jet.Pt > 30]
		Jet_MHT = Jet[np.abs(Jet.eta) < 5]
		Jet_MHT = Jet_MHT[Jet_MHT.PFLooseId > 0.5]
		JetUp_MHT = Jet[Jet.PtTotUncUp > 30]
		JetUp_MHT = JetUp_MHT[np.abs(JetUp_MHT.eta) < 5]
		JetUp_MHT = JetUp_MHT[JetUp_MHT.PFLooseId > 0.5]
		JetDown_MHT = Jet[Jet.PtTotUncDown > 30]
		JetDown_MHT = JetDown_MHT[np.abs(JetDown_MHT.eta) < 5]
		JetDown_MHT = JetDown_MHT[JetDown_MHT.PFLooseId > 0.5]
		Jet_MHT["MHT_x"] = ak.sum(Jet_MHT.Pt*np.cos(Jet_MHT.phi),axis=1,keepdims=False) + ak.sum(JetUp_MHT.PtTotUncUp*np.cos(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.cos(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet_MHT["MHT_y"] = ak.sum(Jet_MHT.Pt*np.sin(Jet_MHT.phi),axis=1,keepdims=False) + ak.sum(JetUp_MHT.PtTotUncUp*np.sin(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.sin(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet_MHT["MHT"] = np.sqrt(Jet_MHT.MHT_x**2 + Jet_MHT.MHT_y**2)
		#print("Jet MHT Defined:")
		
		#HT Seleciton (new)
		#tau_temp1,HT = ak.unzip(ak.cartesian([tau,Jet_MHT], axis = 1, nested = True))
		Jet_HT = Jet[Jet.Pt > 30]
		Jet_HT = Jet_HT[np.abs(Jet_HT.eta) < 3]
		Jet_HT = Jet_HT[Jet_HT.PFLooseId > 0.5]
		JetUp_HT = Jet[Jet.PtTotUncUp > 30]
		JetUp_HT = JetUp_HT[np.abs(JetUp_HT.eta) < 3]
		JetUp_HT = JetUp_HT[JetUp_HT.PFLooseId > 0.5]
		JetDown_HT = Jet[Jet.PtTotUncDown > 30]
		JetUp_HT = JetDown_HT[np.abs(JetDown_HT.eta) < 3]
		JetDown_HT = JetDown_HT[JetDown_HT.PFLooseId > 0.5]
		HT,tau_temp1 = ak.unzip(ak.cartesian([Jet_HT,tau], axis = 1, nested = True))
		HT_up,tau_temp2 = ak.unzip(ak.cartesian([JetUp_HT,tau], axis = 1, nested = True))
		HT_down,tau_temp3 = ak.unzip(ak.cartesian([JetDown_HT,tau], axis = 1, nested = True))
		
		#Get Cross clean free histograms
		HT_Var_NoCrossClean = ak.sum(Jet_HT.Pt,axis = 1,keepdims=True) + ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=True) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=True)
		MET_NoCrossCleaning.fill(ak.ravel(Jet.pfMET))
		MET_Acc_NoCrossClean = hist.accumulators.Mean().fill(ak.ravel(HT_Var_NoCrossClean[HT_Var_NoCrossClean > 0]))
		#HT_NoCrossCleaning.fill(ak.sum(Jet_MHT.Pt,axis = 1,keepdims=False) + ak.sum(JetUp_MHT.PtTotUncUp,axis = 1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown,axis = 1,keepdims=False))	
		HT_NoCrossCleaning.fill(ak.ravel(HT_Var_NoCrossClean[HT_Var_NoCrossClean > 0]))
		HT_num = 0
		for x in ak.sum(Jet.Pt,axis = 1,keepdims=False):
			HT_num += 1
			
		print(ak.sum(Jet.Pt,axis = 1,keepdims=True))
		print("HT Num (No Cross Cleaning): %d"%HT_num)
		HT_Acc_NoCrossClean = hist.accumulators.Mean().fill(ak.ravel(HT_Var_NoCrossClean[HT_Var_NoCrossClean > 0]))
	
		mval_temp = deltaR(tau_temp1,HT) >= 0.5
		print(mval_temp)
		if (len(Jet.Pt) != len(mval_temp)):
			print("Things aren't good")
			if (len(Jet.Pt) > len(mval_temp)):
				print("More Jets than entries in mval_temp")
			if (len(Jet.Pt) < len(mval_temp)):
				print("Fewer entries in Jets than mval_temp")

		Jet_HT["dR"] = mval_temp
		mval_temp = deltaR(tau_temp2,HT_up) >= 0.5
		JetUp_HT["dR"] = mval_temp 
		mval_temp = deltaR(tau_temp3,HT_down) >= 0.5
		JetDown_HT["dR"] = mval_temp

		#print("Pre dR Length %d"%len(Jet))
		Jet_HT = Jet_HT[ak.all(Jet_HT.dR == True, axis = 2)] #Lepton cuts
		JetUp_HT = JetUp_HT[ak.all(JetUp_HT.dR == True, axis = 2)]
		JetDown_HT = JetDown_HT[ak.all(JetDown_HT.dR == True, axis = 2)]
		#print("Post dR Length %d"%len(Jet))
		HT_Val_NoCuts = ak.sum(Jet_HT.Pt,axis = 1,keepdims=True) + ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=True) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=True)
		#Jet["HT"] = ak.sum(Jet.Pt,axis = 1,keepdims=False) #+ ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=False) + ak.sum(JetDown_HT.PtTotUncDown,axis = 1,keepdims=False)
		test_HT = ak.sum(Jet.Pt,axis = 1,keepdims=True)
		HT_num = 0
		print("Test 1:")
		print(ak.sum(Jet.Pt,axis = 1,keepdims=False))
		print("Test 2:")
		print(ak.sum(Jet.Pt,axis = 1,keepdims=True))
		print(ak.ravel(Jet.HT))
		print(Jet.HT)
		for x in ak.ravel(Jet.HT):
			#print(x)
			HT_num += 1
			#if x == 0:
				#print("Anomolous zero (post-cross cleaning)")
		print("HT Num (Cross Cleaning): %d"%HT_num)
		print("HT Len: %d"%len(Jet.HT))
		print("Pt Len: %d"%len(Jet.Pt))
		print("Cross Cleaning Applied")
		print("Len HT = %d"%len(Jet.HT))

		#Histograms of variables relavent to trigger 
		if (self.trigger_bit == 40):
			AK8Jet_Temp = AK8Jet[ak.num(AK8Jet, axis = 1) > 0]
			AK8Pt_NoCut.fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetPt)) #Change to leading pt
			AK8Pt_Acc = hist.accumulators.Mean().fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetPt))
			AK8SoftMass_NoCut.fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetDropMass))
			AK8SoftMass_Acc = hist.accumulators.Mean().fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetDropMass))
		if (self.trigger_bit == 39):
			print("No Cut Accumulators updated")
			HT_NoCut.fill(ak.ravel(HT_Val_NoCuts[HT_Val_NoCuts > 0]))
			HT_Acc_NoCut = hist.accumulators.Mean().fill(ak.ravel(HT_Val_NoCuts[HT_Val_NoCuts > 0]))
			print("HT Mean: %f"%HT_Acc_NoCut.value)
			MET_NoCut.fill(ak.ravel(Jet.pfMET))
			MET_Acc_NoCut = hist.accumulators.Mean().fill(ak.ravel(Jet.pfMET))
			print("MET Mean: %f"%MET_Acc_NoCut.value)
				

		trigger_mask = bit_mask([self.trigger_bit])		
		tau = tau[tau.pt > 30] #pT
		tau = tau[tau.eta < 2.3] #eta
		
		#Loose isolation
		tau = tau[tau.decay >= 0.5]	
		tau = tau[tau.iso >= 0.5]


		#Delta R Cut on taus
		a,b = ak.unzip(ak.cartesian([tau,tau], axis = 1, nested = True))
		mval = deltaR(a,b) < 0.8 
		tau["dRCut"] = mval
		tau = tau[ak.any(tau.dRCut, axis = 2) == True]	
		
		AK8Jet = AK8Jet[(ak.sum(tau.charge,axis=1) == 0)] #Apply charge conservation cut to AK8Jets
		Jet = Jet[(ak.sum(tau.charge,axis=1) == 0)]
		Jet_HT = Jet_HT[(ak.sum(tau.charge,axis=1) == 0)]
		JetUp_HT = JetUp_HT[(ak.sum(tau.charge,axis=1) == 0)]
		JetDown_HT = JetDown_HT[(ak.sum(tau.charge,axis=1) == 0)]
		Jet_MHT = Jet_MHT[(ak.sum(tau.charge,axis=1) == 0)]
		tau = tau[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		
		#Investegate entries with more than 4 taus
		# n_more = len(tau[ak.num(tau) > 4])
		# print("Events with more than 4 taus: %d"%n_more)
		
		# if (n_more > 0):
		# 	print("========!!Important information about events with more than 4 tau!!========")
		# 	diff_num = n_more
		# 	test_obj = tau[ak.num(tau) > 4] 
		# 	n = 5s
		# 	while(n_more > 0):
		# 		N_events = len(test_obj[ak.num(test_obj) == 5])
		# 		print("Number of events with %d taus: %d"%(n,N_events))
		# 		n +=1
		# 		diff_num -= N_events
		

		AK8Jet = AK8Jet[ak.num(tau) >= 4]
		Jet_MHT = Jet_MHT[ak.num(tau) >= 4]
		Jet_HT = Jet_HT[ak.num(tau) >= 4]
		JetUp_HT = JetUp_HT[ak.num(tau) >= 4]
		JetDown_HT = JetDown_HT[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		tau = tau[ak.num(tau) >= 4] #4 tau events
		
		if (self.OrTrigger): #Select for both triggers
			Jet["HT"] = ak.sum(Jet_HT.Pt, axis = 1, keepdims=False) + ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=False) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=False)
			tau = tau[np.bitwise_and(tau.trigger,bit_mask([39,40])) != 0]
			AK8Jet = AK8Jet[np.bitwise_and(AK8Jet.trigger,bit_mask([39,40])) != 0]
			Jet = Jet[np.bitwise_and(Jet.trigger,bit_mask([39,40])) != 0]
			Jet_HT = Jet_HT[np.bitwise_and(Jet_HT.trigger,bit_mask([39,40])) != 0]
			Jet_MHT = Jet_MHT[np.bitwise_and(Jet_MHT.trigger,bit_mask([39,40])) != 0]
		else: #Single Trigger
			if (self.trigger_bit == 40):
				AK8Pt_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetPt))
				AK8Pt_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetPt)
				AK8SoftMass_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetDropMass))
				AK8SoftMass_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetDropMass)
				AK8Pt_all.fill("No Trigger",ak.ravel(AK8Jet.AK8JetPt))
			
			if (self.trigger_bit == 39):
				#Fill Histograms
				HT_Val_PreTrigger = ak.sum(Jet_HT.Pt, axis = 1, keepdims=True) + ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=True) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=True)
				Jet["HT"] = ak.sum(Jet_HT.Pt, axis = 1, keepdims=False) + ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=False) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=False)
				HT_PreTrigg.fill(ak.ravel(HT_Val_PreTrigger[HT_Val_PreTrigger > 0]))
				#HT_NoTrigg_Arr = ak.ravel(HT_Val_PreTrigger[HT_Val_PreTrigger > 0])
				HT_NoTrigg_Arr = ak.ravel(Jet.HT)
				MET_PreTrigg.fill(ak.ravel(Jet.pfMET))
				MET_NoTrigg_Arr = ak.ravel(Jet.pfMET)
				print("MET Len: %d"%len(MET_NoTrigg_Arr))	
				print("HT Len: %d"%len(HT_NoTrigg_Arr))	

			tau = tau[np.bitwise_and(tau.trigger,trigger_mask) == trigger_mask]
			AK8Jet = AK8Jet[np.bitwise_and(AK8Jet.trigger,trigger_mask) == trigger_mask]
			Jet = Jet[np.bitwise_and(Jet.trigger,trigger_mask) == trigger_mask]
			Jet_HT = Jet_HT[np.bitwise_and(Jet_HT.trigger,trigger_mask) == trigger_mask]
			Jet_MHT = Jet_MHT[np.bitwise_and(Jet_MHT.trigger,trigger_mask) == trigger_mask]		

		#Construct all possible valid ditau pairs
		tau_plus = tau[tau.charge > 0]		
		tau_minus = tau[tau.charge < 0]
		tau_plus1, tau_plus2 = ak.unzip(ak.combinations(tau_plus,2))
		tau_minus1, tau_minus2 = ak.unzip(ak.combinations(tau_minus,2))
		
		deltaR11 = deltaR(tau_plus1, tau_minus1)
		deltaR12 = deltaR(tau_plus1, tau_minus2)
		deltaR22 = deltaR(tau_plus2, tau_minus2)
		deltaR21 = deltaR(tau_plus2, tau_minus1)

		#Fill 4tau Mass Histogram
		FourTau_Mass_hist.fill(ak.ravel(four_mass(tau)))
		FourTau_Mass_hist *= (1/FourTau_Mass_hist.sum()) #Normalize the histogram
		FourTau_Mass_Acc = hist.accumulators.Mean().fill()
		
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
            
			AK8Jet_PreTrigger.fill(AK8Pt_NoTrigg_Arr, AK8SoftMass_NoTrigg_Arr)
			AK8Jet_Trigger.fill(AK8Pt_Trigg_Arr, AK8SoftMass_Trigg_Arr)
			eff_AK8Jet = AK8Jet_Trigger/AK8Jet_PreTrigger
		
		if (self.trigger_bit == 39):
			HT_Val_PreTrigger = ak.sum(Jet_HT.Pt, axis = 1, keepdims=True)
			HT_Trigg.fill(ak.ravel(HT_Val_PreTrigger[HT_Val_PreTrigger > 0]))
			#HT_Trigg_Arr = ak.ravel(HT_Val_PreTrigger[HT_Val_PreTrigger > 0])
			HT_Trigg_Arr = ak.ravel(Jet.HT)
			MET_Trigg.fill(ak.ravel(Jet.pfMET))
			MET_Trigg_Arr = ak.ravel(Jet.pfMET)
			pre_triggernum = ak.num(MET_NoTrigg_Arr,axis=0)
			post_triggernum = ak.num(MET_Trigg_Arr,axis=0)	
			
			print("Efficiency (HT+MET Trigger): %f"%(ak.num(MET_Trigg_Arr,axis=0)/ak.num(MET_NoTrigg_Arr,axis=0)))
			Jet_PreTrigger.fill(MET_NoTrigg_Arr, HT_NoTrigg_Arr)
			Jet_Trigger.fill(MET_Trigg_Arr, HT_Trigg_Arr)
			eff_Jet = Jet_Trigger/Jet_PreTrigger
		
		return{
			dataset: {
				"FourTau_Mass_hist": FourTau_Mass_hist
				"FourTau_Mass_Acc": FourTau_Mass_Acc
			}
		}
		# if (self.trigger_bit == 40):
		# 	return{
		# 		 dataset: {
		# 			"AK8JetPt_PreTrigg": AK8Pt_PreTrigg,
		# 			"AK8JetPt_Trigg": AK8Pt_Trigg,
		# 			"AK8JetSoftMass_PreTrigg": AK8SoftMass_PreTrigg,
		# 			"AK8JetSoftMass_Trigg": AK8SoftMass_Trigg,
		# 			"AK8Jet_PreTrigg": AK8Jet_PreTrigger,
		# 			"AK8Jet_Trigg": AK8Jet_Trigger,
		# 			"AK8Jet_eff": eff_AK8Jet,
		# 			"pre_trigger_num": pre_triggernum, 
		# 			"post_trigger_num": post_triggernum,
		# 			"AK8JetPt_NoCut": AK8Pt_NoCut, 
		# 			"AK8JetSoftMass_NoCut": AK8SoftMass_NoCut,
		# 			"Acc_AK8JetPt_NoCut": AK8Pt_Acc,
		# 			"Acc_AK8JetSoftMass_NoCut": AK8SoftMass_Acc
		# 		}
		# 	}
		# if (self.trigger_bit == 39):
		# 	return{
		# 		 dataset: {
		# 			"MET_PreTrigg": MET_PreTrigg,
		# 			"MET_Trigg": MET_Trigg,
		# 			"HT_PreTrigg": HT_PreTrigg,
		# 			"HT_Trigg": HT_Trigg,
		# 			"Jet_PreTrigg": Jet_PreTrigger,
		# 			"Jet_Trigg": Jet_Trigger,
		# 			"Jet_eff": eff_Jet,
		# 			"pre_trigger_num": pre_triggernum,
		# 			"post_trigger_num": post_triggernum,
		# 			"MET_NoCut": MET_NoCut,
		# 			"Acc_MET_NoCut": MET_Acc_NoCut,
		# 			"HT_NoCut": HT_NoCut,
		# 			"Acc_HT_NoCut": HT_Acc_NoCut,
		# 			"MET_NoCrossClean": MET_NoCrossCleaning,
		# 			"Acc_MET_NoCrossClean": MET_Acc_NoCrossClean,
		# 			"HT_NoCrossClean": HT_NoCrossCleaning,
		# 			"Acc_HT_NoCrossClean": HT_Acc_NoCrossClean
		# 		}
		# 	}

	
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
		"AK8JetPt_NoCut" : ["AK8Pt_NoCut_Plot", r"AK8Jet $p_T$ No Cuts"], "AK8JetSoftMass_NoCut" : ["AK8SoftMass_NoCut_Plot", "AK8SoftDrop Mass No Cut"]
	}
	
	trigger_AK8Jet_hist_dict_2d = {
		"AK8Jet_PreTrigg" : ["AK8Jet_PreTriggerHist_Plot", "AK8Jet 2D Histogram No Trigger"], "AK8Jet_Trigg" : ["AK8Jet_TriggerHist_Plot", "AK8Jet 2D Histogram Trigger"],
		"AK8Jet_eff" : ["AK8Jet_Eff_Plot", "AK8Jet 2D Efficiency Histogram Trigger"]
	}

	trigger_MTHTJet_hist_dict_1d = {
		"MET_Trigg" : ["MET_Trigger_Plot","pfMET Trigger"] , "MET_PreTrigg" : ["MET_NoTrigger_Plot","pfMET No Trigger"], "MET_NoCut": ["MET_NoCut_Plot", "pfMET No Cuts/Selections"], 
		"HT_Trigg" : ["HT_Trigger_Plot",r"HT Trigger"], "HT_PreTrigg" : ["HT_NoTrigger_Plot", r"HT No Trigger"], "HT_NoCut" : ["HT_NoCut_Plot", "HT No Cuts/Selections"],
		"MET_NoCrossClean" : ["MET_NoCrossClean_Plot", "pfMET No Cross Cleaning Applied"], "HT_NoCrossClean" : ["HT_NoCrossComp_Plot", "HT No Cross Cleaing Applied"]
	}
	
	trigger_MTHTJet_hist_dict_2d = {
		"Jet_PreTrigg" : ["Jet_PreTriggerHist_Plot", "MET and HT 2D Histogram No Trigger"], "Jet_Trigg" : ["Jet_TriggerHist_Plot", "MET and HT 2D Histogram Trigger"],
		"Jet_eff" : ["Jet_Eff_Plot", "MET and HT Efficiency Histogram Trigger"]
	}
	
	trigger_dict = {"PFHT500_PFMET100_PFMHT100_IDTight": 39, "AK8PFJet400_TrimMass30": 40}

	signal_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v1_Hadd/GluGluToRadionToHHTo4T_M-"
	background_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/"	

	iterative_runner = processor.Runner(
		executor = processor.IterativeExecutor(compression=None),
		schema=BaseSchema
	)


	for mass in mass_str_arr:
		#Grand Unified Background + Signal Dictionary
		file_dict = {"Background" : [background_base + "TTTo2L2Nu.root",background_base + "TTToSemiLeptonic.root",background_base + "TTToHadronic.root", background_base + "ZZ4l.root"], "Signal": [signal_bas + signal_base + mass_str + ".root"]}
		for trigger_name, trigger_bit in trigger_dict.items():
			trigger_out = iterative_runner(file_dict, treename="4tau_tree",processor_instance=FourTauPlotting(trigger_bit))
	
	#Signal
	# for mass_str in mass_str_arr:
	# 	fileName = signal_base + mass_str + ".root"
	# 	events = NanoEventsFactory.from_root(
	# 		fileName,
	# 		treepath="/4tau_tree",
	# 		schemaclass = BaseSchema,
	# 		metadata={"dataset": "boosted_tau"},
	# 	).events()
	# 	p = TauPlotting()
	# 	out = p.process(events)
	# 	#Tau plotting
	# 	for var_name, hist_name_arr in tau_hist_dict.items():
	# 		fig, ax = plt.subplots()
	# 		out["boosted_tau"][var_name].plot1d(ax=ax)
	# 		plt.title(hist_name_arr[1], wrap=True)
	# 		if (hist_name_arr[0] == "AllDitauMass_Plot"):
	# 			ax.legend(title=r"Di-$\tau$ Pair")
	# 		if (hist_name_arr[0] == "AllPt_Plot"):
	# 			ax.legend(title=r"$\tau$")
	# 		plt.savefig(hist_name_arr[0] + "-" + mass_str)
	# 		plt.close()

	# 	#Trigger Plotting
	# 	print("Mass: " + mass_str[0] + "." + mass_str[1] + " TeV")
	# 	for trigger_name, trigger_bit in trigger_dict.items():
	# 		p2 = TriggerStudies(trigger_bit)
	# 		trigger_out = p2.process(events)
	# 		if (trigger_bit == 40):
	# 			trigger_hist_dict_1d = trigger_AK8Jet_hist_dict_1d 
	# 			trigger_hist_dict_2d = trigger_AK8Jet_hist_dict_2d 
	# 		if (trigger_bit == 39):
	# 			trigger_hist_dict_1d = trigger_MTHTJet_hist_dict_1d  
	# 			trigger_hist_dict_2d = trigger_MTHTJet_hist_dict_2d 
			
	# 		for var_name, hist_name_arr in trigger_hist_dict_1d.items():
	# 			fig, ax = plt.subplots()
	# 			trigger_out["boosted_tau"][var_name].plot1d(ax=ax)

	# 			if (hist_name_arr[0][-14:] == "NoTrigger_Plot"):
	# 				plt.title(hist_name_arr[1] + " mass : " + mass_str[0] + " TeV", wrap=True)
	# 			else:
	# 				plt.title(hist_name_arr[1] + " (" + trigger_name + ") , mass : " + mass_str[0] + " TeV", wrap=True)
			
	# 			#Add Text with average and number of entries	
	# 			if (var_name == "MET_NoCut"):
	# 				plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out["boosted_tau"]["Acc_MET_NoCut"].count, transform = ax.transAxes)
	# 				plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out["boosted_tau"]["Acc_MET_NoCut"].value, transform = ax.transAxes)
	# 			if (var_name == "MET_NoCrossClean"):
	# 				plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out["boosted_tau"]["Acc_MET_NoCrossClean"].count, transform = ax.transAxes)
	# 				plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out["boosted_tau"]["Acc_MET_NoCrossClean"].value, transform = ax.transAxes)
	# 			if (var_name == "HT_NoCut"):
	# 				plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out["boosted_tau"]["Acc_HT_NoCut"].count, transform = ax.transAxes)
	# 				plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out["boosted_tau"]["Acc_HT_NoCut"].value, transform = ax.transAxes)
	# 			if (var_name == "HT_NoCrossClean"):
	# 				plt.text(x = 0.14, y = 0.8, s = "Entries: %d"%trigger_out["boosted_tau"]["Acc_HT_NoCrossClean"].count, transform = ax.transAxes)
	# 				plt.text(x = 0.14, y = 0.74, s = "Mean: %.2f GeV"%trigger_out["boosted_tau"]["Acc_HT_NoCrossClean"].value, transform = ax.transAxes)
	# 			if (var_name == "AK8JetPt_NoCut"):
	# 				plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out["boosted_tau"]["Acc_AK8JetPt_NoCut"].count, transform = ax.transAxes)
	# 				plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out["boosted_tau"]["Acc_AK8JetPt_NoCut"].value, transform = ax.transAxes)
	# 			if (var_name == "AK8JetSoftMass_NoCut"):
	# 				plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out["boosted_tau"]["Acc_AK8JetSoftMass_NoCut"].count, transform = ax.transAxes)
	# 				plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out["boosted_tau"]["Acc_AK8JetSoftMass_NoCut"].value, transform = ax.transAxes)
	# 			plt.savefig(hist_name_arr[0] + "-" + mass_str + "-" + trigger_name)
	# 			plt.close()
				  
	# 		for var_name, hist_name_arr in trigger_hist_dict_2d.items():
	# 			fig, ax = plt.subplots()
	# 			trigger_out["boosted_tau"][var_name].plot2d(ax=ax)

	# 			if (hist_name_arr[0][-19:] == "PreTriggerHist_Plot"):
	# 				plt.title(hist_name_arr[1] + " mass : " + mass_str[0] + " TeV", wrap=True)
	# 			else:
	# 				plt.title(hist_name_arr[1] + " (" +  trigger_name + "), mass : " + mass_str[0] + " TeV", wrap=True)
	# 			plt.savefig(hist_name_arr[0] + "-" + mass_str + "-" + trigger_name)
	# 			plt.close()
			
	#Obtain background information
	#background_array = ["ZZ4l",]
	
	#background_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedH3/2018/tt/v2_fast_Hadd/"
	#background_dict = {"ZZ4l" : r"$ZZ \rightarrow 4l$", "top": "Top Background"}
	#file_dict = {"top": [background_base + "TTTo2L2Nu.root",background_base + "TTToSemiLeptonic.root",background_base + "TTToHadronic.root"]}
	#file_dict = {"top": [background_base + "TTTo2L2Nu.root"]} #Single top file


	# for background_name, title in background_dict.items():
	# 	if (background_name == "ZZ4l"):
	# 		events = NanoEventsFactory.from_root(
	# 			background_base + background_name + ".root",
	# 			treepath="/4tau_tree",
	# 			#treepath="/tautau_tree",
	# 			schemaclass = BaseSchema,
	# 			metadata={"dataset": "boosted_tau"},
	# 		).events()
		
	# 	print("Background: " + background_name)	
	# 	for trigger_name, trigger_bit in trigger_dict.items():
	# 		if (background_name == "ZZ4l"):
	# 			p2 = TriggerStudies(trigger_bit, False)
	# 			trigger_out = p2.process(events)
	# 			out_name = "boosted_tau"
	# 		else:
	# 			trigger_out = iterative_runner(file_dict, treename="4tau_tree",processor_instance=TriggerStudies(trigger_bit, False)) 
	# 			out_name = background_name
	# 			#trigger_out = iterative_runner(file_dict, treename="tautau_tree",processor_instance=TriggerStudies(trigger_bit, False)) 
			
	# 		if (trigger_bit == 40):
	# 			trigger_hist_dict_1d = trigger_AK8Jet_hist_dict_1d 
	# 			trigger_hist_dict_2d = trigger_AK8Jet_hist_dict_2d 
			
	# 		if (trigger_bit == 39):
	# 			trigger_hist_dict_1d = trigger_MTHTJet_hist_dict_1d  
	# 			trigger_hist_dict_2d = trigger_MTHTJet_hist_dict_2d 
			
	# 		for var_name, hist_name_arr in trigger_hist_dict_1d.items():
	# 			fig, ax = plt.subplots()
	# 			if (background_name == "ZZ4l"):
	# 				trigger_out["boosted_tau"][var_name].plot1d(ax=ax)
	# 				print("Efficiency = %f"%(trigger_out["boosted_tau"]["post_trigger_num"]/trigger_out["boosted_tau"]["pre_trigger_num"]))
	# 			else:
	# 				#trigger_out[background_name]["boosted_tau"][var_name].plot1d(ax=ax)
	# 				trigger_out[background_name][var_name].plot1d(ax=ax)
	# 				print("Efficiency = %f"%(trigger_out[background_name]["post_trigger_num"]/trigger_out[background_name]["pre_trigger_num"]))
	
	# 			if (hist_name_arr[0][-14:] == "NoTrigger_Plot"):
	# 				plt.title(hist_name_arr[1] + title, wrap=True)
	# 			else:
	# 				plt.title(hist_name_arr[1] + " (" + trigger_name + r"), " + title, wrap=True)
				
	# 			#Add Text with average and number of entries	
	# 			if (var_name == "MET_NoCut"):
	# 				plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out[out_name]["Acc_MET_NoCut"].count, transform = ax.transAxes)
	# 				plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out[out_name]["Acc_MET_NoCut"].value, transform = ax.transAxes)
	# 			if (var_name == "MET_NoCrossClean"):
	# 				plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out[out_name]["Acc_MET_NoCrossClean"].count, transform = ax.transAxes)
	# 				plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out[out_name]["Acc_MET_NoCrossClean"].value, transform = ax.transAxes)
	# 			if (var_name == "HT_NoCut"):
	# 				plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out[out_name]["Acc_HT_NoCut"].count, transform = ax.transAxes)
	# 				plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out[out_name]["Acc_HT_NoCut"].value, transform = ax.transAxes)
	# 			if (var_name == "HT_NoCrossClean"):
	# 				plt.text(x = 0.14, y = 0.8, s = "Entries: %d"%trigger_out[out_name]["Acc_HT_NoCrossClean"].count, transform = ax.transAxes)
	# 				plt.text(x = 0.14, y = 0.74, s = "Mean: %.2f GeV"%trigger_out[out_name]["Acc_HT_NoCrossClean"].value, transform = ax.transAxes)
	# 			if (var_name == "AK8JetPt_NoCut"):
	# 				plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out[out_name]["Acc_AK8JetPt_NoCut"].count, transform = ax.transAxes)
	# 				plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out[out_name]["Acc_AK8JetPt_NoCut"].value, transform = ax.transAxes)
	# 			if (var_name == "AK8JetSoftMass_NoCut"):
	# 				plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out[out_name]["Acc_AK8JetSoftMass_NoCut"].count, transform = ax.transAxes)
	# 				plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out[out_name]["Acc_AK8JetSoftMass_NoCut"].value, transform = ax.transAxes)
				
	# 			plt.savefig(hist_name_arr[0] + "-" + background_name + "-" + trigger_name)
	# 			print(background_name)
	# 			plt.close()
				  
	# 		for var_name, hist_name_arr in trigger_hist_dict_2d.items():
	# 			fig, ax = plt.subplots()
	# 			if (background_name == "ZZ4l"):
	# 				trigger_out["boosted_tau"][var_name].plot2d(ax=ax)
	# 			else:
	# 				#trigger_out[background_name]["boosted_tau"][var_name].plot2d(ax=ax)
	# 				if ("eff" not in var_name):
	# 					trigger_out[background_name][var_name].plot2d(ax=ax)
	# 				if (var_name == "Jet_eff"):
	# 					#Set up efficiency histogram
	# 					eff_Jet = hist.Hist(
	# 						hist.axis.Regular(20, 0, 1200., name = "pfMET" , label = r"MET [GeV]"),
	# 						hist.axis.Regular(20, 0, 3500., name = "HT", label = r"HT [GeV]")
	# 					)
	# 					eff_Jet = trigger_out[background_name]["Jet_Trigg"]/trigger_out[background_name]["Jet_PreTrigg"] 
	# 					eff_Jet.plot2d(ax=ax)
	# 				if (var_name == "AK8Jet_eff"):
	# 					print("AK8Jet Stuff!!")
	# 					eff_AK8Jet = hist.Hist(
	# 						hist.axis.Regular(20, 0, 1100, name="JetPt", label=r"AK8Jet $p_T$ [GeV]"),
	# 						hist.axis.Regular(10, 0, 300, name="SoftMass", label="AK8Jet Soft Mass [GeV]")
	# 					)		
	# 					eff_AK8Jet = trigger_out[background_name]["AK8Jet_Trigg"]/trigger_out[background_name]["AK8Jet_PreTrigg"]		
	# 					eff_AK8Jet.plot2d(ax=ax)
	
	# 			if (hist_name_arr[0][-19:] == "PreTriggerHist_Plot"):
	# 				plt.title(hist_name_arr[1] + title, wrap=True)
	# 			else:
	# 				plt.title(hist_name_arr[1] + trigger_name + "), " + title, wrap=True)
	# 			plt.savefig(hist_name_arr[0] + "-" + background_name + "-" + trigger_name)
	# 			plt.close()
		
