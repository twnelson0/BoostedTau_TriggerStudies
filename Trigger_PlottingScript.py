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

def pdf(x,a=1,x0=0):
	return (1 + np.exp(a/(x-x0)))**-1

def delta_phi(vec1,vec2):
	return (vec1.phi - vec2.phi + pi) % (2*pi) - pi	

def deltaR(part1, part2):
	return np.sqrt((part2.eta - part1.eta)**2 + (delta_phi(part2,part1))**2)

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
				"eta": events.jetEta,
				"phi": events.jetPhi,
				"trigger": events.HLTJet,
			},
			with_name="PFJetArray",
			behavior=candidate.behavior,
		)
		
		#Histograms (AK8Jet) (Trigger bit = 40)
		AK8Pt_all = hist.Hist.new.StrCat(["No Trigger","Trigger"], name = "AK8Pt_hist").Reg(40,0,1600, name="AK8Pt", label = "AK8 Jet r$p_T$ [GeV]").Double()	
		AK8SoftMass_all = hist.Hist.new.StrCat(["No Trigger","Trigger"], name = "AK8SoftMass_hist").Reg(40,0,400, name="AK8SoftMass", label = "AK8 Jet Soft Drop Mass [GeV]").Double()
		AK8Pt_PreTrigg = hist.Hist.new.Reg(40, 0, 1500, name = "JetPt_Trigg", label = r"AK8Jet $p_T$ [GeV]").Double()
		AK8Pt_NoCut = hist.Hist.new.Reg(40, 0, 1500, name = "JetPt_NoCut", label = r"AK8Jet $p_T$ [GeV]").Double()
		AK8Pt_Trigg = hist.Hist.new.Reg(40, 0, 1500, name = "JetPt_Trigg", label = r"AK8Jet $p_T$ [GeV]").Double()
		AK8Pt_TurnOn = hist.Hist.new.Reg(40, 0, 1500, name = "JetPt_TO", label = r"AK8Jet $p_T$ [GeV]").Double()
		AK8Pt_Debug = hist.Hist.new.Reg(40, 0, 1500, name = "JetPt_Debug", label = r"AK8Jet $p_T$ [GeV]").Double()
		AK8Eta_Debug = hist.Hist.new.Reg(40, -4, 4, name = "JetEta_Debug", label = r"AK8Jet $\eta$").Double()
		AK8SoftMass_PreTrigg = hist.Hist.new.Reg(50, 0, 250, name = "SoftMass_Trigg", label = "AK8Jet Soft Mass [GeV]").Double()
		AK8SoftMass_NoCut = hist.Hist.new.Reg(50, 0, 250, name = "SoftMass_NoCut", label = "AK8Jet Soft Mass [GeV]").Double()
		AK8SoftMass_Trigg = hist.Hist.new.Reg(50, 0, 250, name = "SoftMass_Trigg", label = "AK8Jet Soft Mass [GeV]").Double()		
		AK8SoftMass_TurnOn = hist.Hist.new.Reg(50, 0, 250, name = "SoftMass_TO", label = "AK8Jet Soft Mass [GeV]").Double()		
		AK8JetMult_NoCut = hist.Hist.new.Reg(8,0,7, name = "AK8 Jet Multiplicity", label = "AK8 Jet Multiplicity").Double()
		AK8JetMult_PreTrigg = hist.Hist.new.Reg(8,0,7, name = "AK8 Jet Multiplicity", label = "AK8 Jet Multiplicity").Double()
		AK8JetMult_Trigg = hist.Hist.new.Reg(8,0,7, name = "AK8 Jet Multiplicity", label = "AK8 Jet Multiplicity").Double()
		
		#2D Histograms
		AK8Jet_PreTrigger = hist.Hist(
						hist.axis.Regular(50, 0, 1500, name="JetPt", label=r"AK8Jet $p_T$ [GeV]"),
						hist.axis.Regular(25, 0, 250, name="SoftMass", label=r"AK8Jet Soft Mass [GeV]")
					)		
		AK8Jet_Trigger = hist.Hist(
						hist.axis.Regular(50, 0, 1500, name="JetPt", label=r"AK8Jet $p_T$ [GeV]"),
						hist.axis.Regular(25, 0, 250, name="SoftMass", label="AK8Jet Soft Mass [GeV]")
					)		
		eff_AK8Jet = hist.Hist(
						hist.axis.Regular(50, 0, 1500, name="JetPt", label=r"AK8Jet $p_T$ [GeV]"),
						hist.axis.Regular(25, 0, 250, name="SoftMass", label="AK8Jet Soft Mass [GeV]")
					)		

		#Histograms (MET and HT) (Trigger bit = 27)
		MET_PreTrigg = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()
		MET_NoCut = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()
		MET_NoCrossCleaning = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()
		MET_Trigg = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()
		MET_TurnOn = hist.Hist.new.Reg(30, 0, 1200., name="MET", label="MET [GeV]").Double()
		MHT_PreTrigg = hist.Hist.new.Reg(30, 0, 1200., name="MHT", label="MHT [GeV]").Double()
		MHT_NoCut = hist.Hist.new.Reg(30, 0, 1200., name="MHT", label="MHT [GeV]").Double()
		MHT_NoCrossCleaning = hist.Hist.new.Reg(30, 0, 1200., name="MHT", label="MHT [GeV]").Double()
		MHT_Trigg = hist.Hist.new.Reg(30, 0, 1200., name="MHT", label="MHT [GeV]").Double()
		MHT_TurnOn = hist.Hist.new.Reg(30, 0, 1200., name="MHT", label="MHT [GeV]").Double()

		#2D Histograms
		Jet_PreTrigger = hist.Hist(
			hist.axis.Regular(20, 0, 1000., name = "pfMET [GeV]" , label = r"pfMET [GeV]"),
			hist.axis.Regular(10, 0, 1000., name = "MHT", label = r"MHT [GeV]")
		)
		Jet_Trigger = hist.Hist(
			hist.axis.Regular(20, 0, 1000., name = "pfMET [GeV]" , label = r"pfMET [GeV]"),
			hist.axis.Regular(10, 0, 1000., name = "MHT", label = r"MHT [GeV]")
		)
		eff_Jet = hist.Hist(
			hist.axis.Regular(20, 0, 1000., name = "pfMET [GeV]" , label = r"pfMET [GeV]"),
			hist.axis.Regular(10, 0, 1000., name = "MHT", label = r"MHT [GeV]")
		)

		
		#MHT
		Jet_MHT = Jet[Jet.Pt > 30]
		Jet_MHT = Jet_MHT[np.abs(Jet_MHT.eta) < 5]
		Jet_MHT = Jet_MHT[Jet_MHT.PFLooseId > 0.5]
		Jet_MHT["MHT_x"] = ak.sum(Jet_MHT.Pt*np.cos(Jet_MHT.phi),axis=1,keepdims=False) #+ ak.sum(JetUp_MHT.PtTotUncUp*np.cos(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.cos(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet_MHT["MHT_y"] = ak.sum(Jet_MHT.Pt*np.sin(Jet_MHT.phi),axis=1,keepdims=False) #+ ak.sum(JetUp_MHT.PtTotUncUp*np.sin(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.sin(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet_MHT["MHT"] = np.sqrt(Jet_MHT.MHT_x**2 + Jet_MHT.MHT_y**2)
		MHT_NoCrossCleaning.fill(ak.ravel(Jet_MHT.MHT))
		MHT_Acc_NoCrossingClean = hist.accumulators.Mean().fill(ak.ravel(Jet_MHT.MHT))
		print("Jet MHT Defined:")
	
		#mval_temp = deltaR(tau_temp1,HT) >= 0.5
		#print(mval_temp)
		if (len(Jet.Pt) != len(mval_temp)):
			print("Things aren't good")
			if (len(Jet.Pt) > len(mval_temp)):
				print("More Jets than entries in mval_temp")
			if (len(Jet.Pt) < len(mval_temp)):
				print("Fewer entries in Jets than mval_temp")

		#Histograms of variables relavent to trigger 
		if (self.trigger_bit == 40):
			AK8Jet_Temp = AK8Jet[ak.num(AK8Jet, axis = 1) > 0]
			AK8Pt_NoCut.fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetPt)) #Change to leading pt
			AK8Pt_Acc = hist.accumulators.Mean().fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetPt))
			AK8SoftMass_NoCut.fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetDropMass))
			AK8SoftMass_Acc = hist.accumulators.Mean().fill(ak.ravel(AK8Jet_Temp[:,0].AK8JetDropMass))
			AK8JetMult_NoCut.fill(ak.num(AK8Jet, axis=1))
			
		if (self.trigger_bit == 27):
			print("No Cut Accumulators updated")
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
		Jet_MHT = Jet_MHT[(ak.sum(tau.charge,axis=1) == 0)]
		tau = tau[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation

		AK8Jet = AK8Jet[ak.num(tau) >= 4]
		Electron = Electron[ak.num(tau) >= 4]
		Muon = Muon[ak.num(tau) >= 4]
		Jet_MHT = Jet_MHT[ak.num(tau) >= 4]
		Jet_HT = Jet_HT[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		tau = tau[ak.num(tau) >= 4] #4 tau events
		

		if (self.trigger_bit == 40):
			AK8JetMult_PreTrigg.fill(ak.num(AK8Jet,axis=1))
			AK8Pt_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetPt))
			AK8Pt_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetPt)
			AK8SoftMass_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetDropMass))
			AK8SoftMass_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetDropMass)
			AK8Pt_all.fill("No Trigger",ak.ravel(AK8Jet.AK8JetPt))
			AK8Pt_Debug.fill(ak.ravel(AK8Jet[AK8Jet.AK8JetDropMass <= 10].AK8JetPt))
			AK8Eta_Debug.fill(ak.ravel(AK8Jet[AK8Jet.AK8JetDropMass <= 10].eta))
		
		if (self.trigger_bit == 27):
			#Fill Histograms
			HT_Val_PreTrigger = ak.sum(Jet_HT.Pt, axis = 1, keepdims=True) #+ ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=True) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=True)
			Jet["MHT_x"] = ak.sum(Jet_MHT.Pt*np.cos(Jet_MHT.phi),axis=1,keepdims=False)
			Jet["MHT_y"] = ak.sum(Jet_MHT.Pt*np.sin(Jet_MHT.phi),axis=1,keepdims=False)
			Jet["MHT"] = np.sqrt(Jet.MHT_x**2 + Jet.MHT_y**2)
			MET_PreTrigg.fill(ak.ravel(Jet.pfMET))
			MHT_PreTrigg.fill(ak.ravel(Jet_MHT.MHT))
			MET_NoTrigg_Arr = ak.ravel(Jet.pfMET)
			MHT_NoTrigg_Arr = ak.ravel(Jet.MHT)
			print("MET Len: %d"%len(MET_NoTrigg_Arr))	
			print("MHT Len: %d"%len(MHT_NoTrigg_Arr))	

		#Apply Online trigger
		tau = tau[np.bitwise_and(tau.trigger,trigger_mask) == trigger_mask]
		AK8Jet = AK8Jet[np.bitwise_and(AK8Jet.trigger,trigger_mask) == trigger_mask]
		Jet = Jet[np.bitwise_and(Jet.trigger,trigger_mask) == trigger_mask]
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
			#AK8Jet = AK8Jet[ak.num(AK8Jet,axis=1) > 0] #Remove 0 multiplicity events
			AK8Pt_Trigg.fill(ak.ravel(AK8Jet.AK8JetPt))
			AK8Pt_Test = ak.ravel(AK8Jet[AK8Jet.AK8JetDropMass < 10].AK8JetPt)
			print("Number of Jets with Soft Drop Mass < 10 GeV = %d"%len(AK8Pt_Test))
			AK8SoftMass_Trigg.fill(ak.ravel(AK8Jet.AK8JetDropMass))
			AK8Pt_all.fill("Trigger",ak.ravel(AK8Jet.AK8JetPt))	
			AK8Pt_Trigg_Arr = ak.ravel(AK8Jet.AK8JetPt)
			AK8SoftMass_all.fill("Trigger",ak.ravel(AK8Jet.AK8JetDropMass))	
			AK8SoftMass_Trigg_Arr = ak.ravel(AK8Jet.AK8JetDropMass)
			pre_triggernum = ak.num(AK8Pt_NoTrigg_Arr,axis=0)
			post_triggernum = ak.num(AK8Pt_Trigg_Arr,axis=0)
			AK8JetMult_Trigg.fill(ak.num(AK8Jet,axis=1))
			if (len(AK8Pt_Trigg_Arr) == 0):
				print("!!=================NO AK8 Jet PT========================!!")
			
			#Make 1d Turn on Plots
			AK8Pt_TurnOn = AK8Pt_Trigg/AK8Pt_PreTrigg
			AK8Pt_ErrorBars = hist.intervals.ratio_uncertainty(AK8Pt_Trigg.view(), AK8Pt_PreTrigg.view(), uncertainty_type = "efficiency") 
			AK8SoftMass_TurnOn = AK8SoftMass_Trigg/AK8SoftMass_PreTrigg
			AK8SoftMass_ErrorBars = hist.intervals.ratio_uncertainty(AK8SoftMass_Trigg.view(), AK8SoftMass_PreTrigg.view(), uncertainty_type = "efficiency") 
	
			if (self.signal):
				print("Efficiency (AK8Jet Trigger): %f"%(ak.num(AK8Pt_Trigg_Arr,axis=0)/ak.num(AK8Pt_NoTrigg_Arr,axis=0)))
			AK8Jet_PreTrigger.fill(AK8Pt_NoTrigg_Arr, AK8SoftMass_NoTrigg_Arr)
			AK8Jet_Trigger.fill(AK8Pt_Trigg_Arr, AK8SoftMass_Trigg_Arr)
			eff_AK8Jet = AK8Jet_Trigger/AK8Jet_PreTrigger
		
		if (self.trigger_bit == 27):
			MET_Trigg.fill(ak.ravel(Jet.pfMET))
			MHT_Trigg.fill(ak.ravel(Jet_MHT.MHT))
			MET_Trigg_Arr = ak.ravel(Jet.pfMET)
			MHT_Trigg_Arr = ak.ravel(Jet.MHT)
			pre_triggernum = ak.num(MET_NoTrigg_Arr,axis=0)
			post_triggernum = ak.num(MET_Trigg_Arr,axis=0)

			#1-d turn on plots
			MHT_TurnOn = MHT_Trigg/MHT_PreTrigg	
			MHT_ErrorBars = hist.intervals.ratio_uncertainty(MHT_Trigg.view(), MHT_PreTrigg.view(), uncertainty_type = "efficiency")	
			MET_TurnOn = MET_Trigg/MET_PreTrigg	
			MET_ErrorBars = hist.intervals.ratio_uncertainty(MET_Trigg.view(), MET_PreTrigg.view(), uncertainty_type = "efficiency")	
			
			print("Efficiency (HT+MET Trigger): %f"%(ak.num(MET_Trigg_Arr,axis=0)/ak.num(MET_NoTrigg_Arr,axis=0)))
			Jet_PreTrigger.fill(MET_NoTrigg_Arr + MHT_NoTrigg_Arr, HT_NoTrigg_Arr)
			Jet_Trigger.fill(MET_Trigg_Arr + MHT_Trigg_Arr, HT_Trigg_Arr)
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
					"AK8JetMult_Trigg": AK8JetMult_Trigg,
					"AK8JetPt_TurnOn": AK8Pt_TurnOn,
					"AK8JetSoftMass_TurnOn": AK8SoftMass_TurnOn,
					"AK8JetPt_ErrorBars": AK8Pt_ErrorBars,
					"AK8JetSoftMass_ErrorBars": AK8SoftMass_ErrorBars,
					"AK8JetPt_Debug": AK8Pt_Debug,
					"AK8JetEta_Debug": AK8Eta_Debug
				}
			}
		if (self.trigger_bit == 27):
			return{
				 dataset: {
					"MET_PreTrigg": MET_PreTrigg,
					"MHT_PreTrigg": MHT_PreTrigg,
					"MET_Trigg": MET_Trigg,
					"MHT_Trigg": MHT_Trigg,
					"Jet_PreTrigg": Jet_PreTrigger,
					"Jet_Trigg": Jet_Trigger,
					"Jet_eff": eff_Jet,
					"pre_trigger_num": pre_triggernum,
					"post_trigger_num": post_triggernum,
					"MET_NoCut": MET_NoCut,
					"Acc_MET_NoCut": MET_Acc_NoCut,
					"Acc_HT_NoCut": HT_Acc_NoCut,
					"MET_NoCrossClean": MET_NoCrossCleaning,
					"MHT_NoCrossClean": MHT_NoCrossCleaning,
					"Acc_MET_NoCrossClean": MET_Acc_NoCrossClean,
					"Acc_MHT_NoCrossClean": MHT_Acc_NoCrossingClean,
					"Acc_HT_NoCrossClean": HT_Acc_NoCrossClean,
					"Acc_HT_PreTrigg": HT_Acc_PreTrigg,
					"Acc_HT_Trigg": HT_Acc_Trigg,
					"MHT_TurnOn" : MHT_TurnOn,
					"MET_TurnOn" : MET_TurnOn,
					"MHT_ErrorBars" : MHT_ErrorBars,
					"MET_ErrorBars" : MET_ErrorBars
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
		
		tau = tau[ak.num(tau) >= 4] #4 tau events (unsure about this)	
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
		lead_cond1 = np.bitwise_and(tau_plus1.pt > tau_minus1.pt, deltaR11 < deltaR12)
		lead_cond2 = np.bitwise_and(tau_plus1.pt > tau_minus1.pt, deltaR12 < deltaR11)
		lead_cond3 = np.bitwise_and(tau_plus1.pt < tau_minus1.pt, deltaR22 < deltaR21)
		lead_cond4 = np.bitwise_and(tau_plus1.pt < tau_minus1.pt, deltaR21 < deltaR22)
		
		dimass_all_hist.fill("Leading pair", ak.ravel(di_mass(tau_plus1[lead_cond1], tau_minus1[lead_cond1])))
		dimass_all_hist.fill("Leading pair", ak.ravel(di_mass(tau_plus2[lead_cond2], tau_minus1[lead_cond2])))
		dimass_leading_acc = hist.accumulators.Mean().fill(ak.concatenate([ak.ravel(di_mass(tau_plus1[lead_cond1], tau_minus1[lead_cond1])), ak.ravel(di_mass(tau_plus2[lead_cond2], tau_minus1[lead_cond2]))]))
		dimass_all_hist.fill("Subleading pair", ak.ravel(di_mass(tau_plus1[lead_cond4], tau_minus2[lead_cond4])))
		dimass_all_hist.fill("Subleading pair", ak.ravel(di_mass(tau_plus2[lead_cond3], tau_minus2[lead_cond3])))
		
		dimass_subleading_acc = hist.accumulators.Mean().fill(ak.concatenate([ak.ravel(di_mass(tau_plus1[lead_cond4], tau_minus2[lead_cond4])), ak.ravel(di_mass(tau_plus2[lead_cond3], tau_minus2[lead_cond3]))]))
		dimass_all_hist *= 1/(dimass_all_hist.sum())		
		
		ditau_mass1_hist.fill(ak.ravel(di_mass(tau_plus1[lead_cond1], tau_minus1[lead_cond1])))	
		ditau_mass1_hist.fill(ak.ravel(di_mass(tau_plus1[lead_cond2], tau_minus2[lead_cond2])))
		ditau_mass1_hist *= (1/ditau_mass1_hist.sum())
		ditau_mass2_hist.fill(ak.ravel(di_mass(tau_plus2[lead_cond3], tau_minus2[lead_cond3])))	
		ditau_mass2_hist.fill(ak.ravel(di_mass(tau_plus2[lead_cond4], tau_minus1[lead_cond4])))
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
	#trigger_dict = {"PFHT500_PFMET100_PFMHT100_IDTight": [27], "AK8PFJet400_TrimMass30": [40]}

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
		"AK8JetMult_NoCut" : ["AK8JetMult_NoCut_Plot", "AK8Jet Multiplicity"], "AK8JetPt_TurnOn" : ["AK8JetPt_TurnOn_Plot","AK8 Jet $p_T$ Turn-on Plot"], 
		"AK8JetSoftMass_TurnOn" : ["AK8JetSoftMass_TurnOn", "AK8 Jet Soft Drop Mass Turn-on Plot"], "AK8JetPt_Debug" : ["AK8JetPt_Debug", "AK8Jet $p_T$ dubug plot, no trigger applied"],
		"AK8JetEta_Debug" : ["AK8JetEta_Debug", "AK8Jet $\eta$ dubug plot, no trigger applied"]
	}
	
	trigger_AK8Jet_hist_dict_2d = {
		"AK8Jet_PreTrigg" : ["AK8Jet_PreTriggerHist_Plot", "AK8Jet 2D Histogram No Trigger"], "AK8Jet_Trigg" : ["AK8Jet_TriggerHist_Plot", "AK8Jet 2D Histogram Trigger"],
		"AK8Jet_eff" : ["AK8Jet_Eff_Plot", "AK8Jet 2D Efficiency Histogram Trigger"]
	}

	trigger_MTHTJet_hist_dict_1d = {
		"MET_Trigg" : ["MET_Trigger_Plot","pfMET Trigger"] , "MET_PreTrigg" : ["MET_NoTrigger_Plot","pfMET No Trigger"], "MET_NoCut": ["MET_NoCut_Plot", "pfMET No Cuts/Selections"], 
		"MHT_Trigg" : ["MHT_Trigger_Plot","MHT Trigger"] , "MHT_PreTrigg" : ["MHT_NoTrigger_Plot","MHT No Trigger"],
		"MET_NoCrossClean" : ["MET_NoCrossClean_Plot", "pfMET No Cross Cleaning Applied"], "MHT_NoCrossClean" : ["MHT_NoCrossClean_Plot", "MHT No Cross Cleaning Applied"], 
		"MHT_TurnOn" : ["MHT_TurnOn_Plot","Jet MHT Turn-on Plot"], "MET_TurnOn" : ["MET_TurnOn_Plot", "Jet pfMET Turn-on Plot"]
	}
	
	trigger_MTHTJet_hist_dict_2d = {
		"Jet_PreTrigg" : ["Jet_PreTriggerHist_Plot", "MET and MHT 2D Histogram No Trigger"], "Jet_Trigg" : ["Jet_TriggerHist_Plot", "MET and MHT 2D Histogram Trigger"],
		"Jet_eff" : ["Jet_Eff_Plot", "MET and MHT Efficiency Histogram Trigger"]
	}

	errorBar_dict = {"AK8JetPt_TurnOn": "AK8JetPt_ErrorBars", "AK8JetSoftMass_TurnOn":"AK8JetSoftMass_ErrorBars", "MET_TurnOn" :"MET_ErrorBars", "MHT_TurnOn" :"MHT_ErrorBars"}
	
	trigger_dict = {"PFMET120_PFMHT120_IDTight": 27, "AK8PFJet400_TrimMass30": 40}

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
			if (trigger_bit == 27):
				trigger_hist_dict_1d = trigger_MTHTJet_hist_dict_1d  
				trigger_hist_dict_2d = trigger_MTHTJet_hist_dict_2d 
			
			for var_name, hist_name_arr in trigger_hist_dict_1d.items():
				fig, ax = plt.subplots()
				if (var_name[-6:] == "TurnOn"):
					#hist.plot.histplot(trigger_out["boosted_tau"][var_name])
					hist.plot.histplot(trigger_out["boosted_tau"][var_name],histtype="errorbar", yerr = trigger_out["boosted_tau"][errorBar_dict[var_name]])
					plt.ylabel("Efficiency")
					#plt.plot(trigger_out["boosted_tau"][var_name].axes[0].centers, trigger_out["boosted_tau"][var_name].values(), 'o')
					#trigger_out["boosted_tau"][var_name].plot1d(ax=ax).centers
					#trigger_out["boosted_tau"][var_name].plot_pull(pdf)
					#trigger_out["boosted_tau"][var_name].plot1d(ax=ax)
				else:
					trigger_out["boosted_tau"][var_name].plot1d(ax=ax)

				if (hist_name_arr[0][-14:] == "NoTrigger_Plot"):
					plt.title(hist_name_arr[1] + " mass : " + mass_str[0] + " TeV", wrap=True)
				else:
					plt.title(hist_name_arr[1] + " (" + trigger_name + ")\n mass : " + mass_str[0] + " TeV", wrap=True)
			
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
				if (var_name == "HT_PreTrigg"):
					plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out["boosted_tau"]["Acc_HT_PreTrigg"].count, transform = ax.transAxes)
					plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out["boosted_tau"]["Acc_HT_PreTrigg"].value, transform = ax.transAxes)
				if (var_name == "HT_Trigg"):
					plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out["boosted_tau"]["Acc_HT_Trigg"].count, transform = ax.transAxes)
					plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out["boosted_tau"]["Acc_HT_Trigg"].value, transform = ax.transAxes)
				plt.savefig(hist_name_arr[0] + "-" + mass_str + "-" + trigger_name)
				plt.close()
				  
			for var_name, hist_name_arr in trigger_hist_dict_2d.items():
				fig, ax = plt.subplots()
				trigger_out["boosted_tau"][var_name].plot2d(ax=ax)

				if (hist_name_arr[0][-19:] == "PreTriggerHist_Plot"):
					plt.title(hist_name_arr[1] + " mass : " + mass_str[0] + " TeV", wrap=True)
				else:
					plt.title(hist_name_arr[1] + " (" +  trigger_name + ")\n mass : " + mass_str[0] + " TeV", wrap=True)
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
			
			if (trigger_bit == 27):
				trigger_hist_dict_1d = trigger_MTHTJet_hist_dict_1d  
				trigger_hist_dict_2d = trigger_MTHTJet_hist_dict_2d 
			
			for var_name, hist_name_arr in trigger_hist_dict_1d.items():
				fig, ax = plt.subplots()
				if (background_name == "ZZ4l"):
					if (var_name[-6:] == "TurnOn"):
						#hist.plot.histplot(trigger_out["boosted_tau"][var_name])
						hist.plot.histplot(trigger_out["boosted_tau"][var_name],histtype="errorbar", yerr = trigger_out["boosted_tau"][errorBar_dict[var_name]])
						plt.ylabel("Efficiency")
						#trigger_out["boosted_tau"][var_name].plot1d(ax=ax)
					else:
						trigger_out["boosted_tau"][var_name].plot1d(ax=ax)
					#trigger_out["boosted_tau"][var_name].plot1d(ax=ax)
					print("Efficiency = %f"%(trigger_out["boosted_tau"]["post_trigger_num"]/trigger_out["boosted_tau"]["pre_trigger_num"]))
				else:
					#trigger_out[background_name]["boosted_tau"][var_name].plot1d(ax=ax)
					if (var_name[-6:] == "TurnOn"):
						#hist.plot.histplot(trigger_out[background_name][var_name])
						hist.plot.histplot(trigger_out[background_name][var_name],histtype="errorbar", yerr = trigger_out[background_name][errorBar_dict[var_name]])
						plt.ylabel("Efficiency")
						#trigger_out[background_name][var_name].plot1d(ax=ax)
					else:
						trigger_out[background_name][var_name].plot1d(ax=ax)
					
					#trigger_out[background_name][var_name].plot1d(ax=ax)
					print("Efficiency = %f"%(trigger_out[background_name]["post_trigger_num"]/trigger_out[background_name]["pre_trigger_num"]))
	
				if (hist_name_arr[0][-14:] == "NoTrigger_Plot"):
					plt.title(hist_name_arr[1] + title, wrap=True)
				else:
					plt.title(hist_name_arr[1] + "\n(" + trigger_name + ") " + title, wrap=True)
				
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
				if (var_name == "HT_PreTrigg"):
					plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out[out_name]["Acc_HT_PreTrigg"].count, transform = ax.transAxes)
					plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out[out_name]["Acc_HT_PreTrigg"].value, transform = ax.transAxes)
				if (var_name == "HT_Trigg"):
					plt.text(x = 0.74, y = 0.8, s = "Entries: %d"%trigger_out[out_name]["Acc_HT_Trigg"].count, transform = ax.transAxes)
					plt.text(x = 0.74, y = 0.74, s = "Mean: %.2f GeV"%trigger_out[out_name]["Acc_HT_Trigg"].value, transform = ax.transAxes)
				
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
					plt.title(hist_name_arr[1] + "\n(" +  trigger_name + ") " + title, wrap=True)
				plt.savefig(hist_name_arr[0] + "-" + background_name + "-" + trigger_name)
				plt.close()
		
