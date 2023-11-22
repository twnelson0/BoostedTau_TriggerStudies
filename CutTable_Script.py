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
	file.write("Sample File(s) & PFHT500\\_PFMET100 PFMHT100\\_IDTight Efficiency & AK8PFJet400\\_TrimMass30 Efficiency \\\\ \n")
	file.write("\\hline \n")
	
	#Fill table
	for sample, eff_arr in table_dict.items():
		file.write(sample + " & " + "%.3f"%eff_arr[0] + " & " + "%.3f"%eff_arr[1] + "\\\\")
		file.write("\n")
	file.write("\\hline \n")
	file.write("\\end{tabular}")
	file.write("\\end{document}")
	file.close()


def TriggerDebuggTable(table_title,  table_dict):
	file_name = "Efficiency_Table_PFTrimMass.tex"
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
	file.write("Sample File(s) & PFJet400 & TrimMass30 \\\\ \n")
	file.write("\\hline \n")
	
	#Fill table
	for sample, eff_arr in table_dict.items():
		file.write(sample + " & " + "%.3f"%eff_arr[0] + " & " + "%.3f"%eff_arr[1] + "\\\\")
		file.write("\n")
	file.write("\\hline \n")
	file.write("\\end{tabular}")
	file.write("\\end{document}")
	file.close() 
	

#class EffTable()

class TriggerStudies(processor.ProcessorABC):
	def __init__(self, trigger_bit, trigger_cut = True, offline_cut = False, signal = True, cut_num = 0):
		self.trigger_bit = trigger_bit
		self.signal = signal
		self.offline_cut = offline_cut
		self.trigger_cut = trigger_cut
		self.cut_num = cut_num
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
		CutFlow_Table = hist.Hist.new.Reg(9,0,8, name = "CutFlow", label = "Cut").Int64()
		CutFlow_Table.fill(0*np.ones([len(ak.ravel(tau.pt))]))
		
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
		
		#Set up variables for offline cuts#MHT
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
		#MET_NoCrossCleaning.fill(ak.ravel(Jet.pfMET))
		MET_Acc_NoCrossClean = hist.accumulators.Mean().fill(ak.ravel(HT_Var_NoCrossClean[HT_Var_NoCrossClean > 0]))
		#HT_NoCrossCleaning.fill(ak.sum(Jet_MHT.Pt,axis = 1,keepdims=False)) #+ ak.sum(JetUp_MHT.PtTotUncUp,axis = 1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown,axis = 1,keepdims=False))	
		#HT_NoCrossCleaning.fill(ak.ravel(HT_Var_NoCrossClean[HT_Var_NoCrossClean > 0]))
		#HT_num = 0
		#for x in ak.sum(Jet.Pt,axis = 1,keepdims=False):
		#	HT_num += 1
			
		#print(ak.sum(Jet.Pt,axis = 1,keepdims=True))
		#print("HT Num (No Cross Cleaning): %d"%HT_num)
		HT_Acc_NoCrossClean = hist.accumulators.Mean().fill(ak.ravel(HT_Var_NoCrossClean[HT_Var_NoCrossClean > 0]))
	
		mval_temp = deltaR(tau_temp1,HT) >= 0.5
		#print(mval_temp)
		#if (len(Jet.Pt) != len(mval_temp)):
		#	print("Things aren't good")
		#	if (len(Jet.Pt) > len(mval_temp)):
		#		print("More Jets than entries in mval_temp")
		#	if (len(Jet.Pt) < len(mval_temp)):
		#		print("Fewer entries in Jets than mval_temp")

		Jet_HT["dR"] = mval_temp
		#mval_temp = deltaR(tau_temp2,HT_up) >= 0.5
		#JetUp_HT["dR"] = mval_temp 
		#mval_temp = deltaR(tau_temp3,HT_down) >= 0.5
		#JetDown_HT["dR"] = mval_temp

		#print("Pre dR Length %d"%len(Jet))
		#Jet_HT = Jet_HT[ak.all(Jet_HT.dR == True, axis = 2)] #Lepton cuts
		#JetUp_HT = JetUp_HT[ak.all(JetUp_HT.dR == True, axis = 2)]
		#JetDown_HT = JetDown_HT[ak.all(JetDown_HT.dR == True, axis = 2)]
		#print("Post dR Length %d"%len(Jet))
		HT_Val_NoCuts = ak.sum(Jet_HT.Pt,axis = 1,keepdims=True) #+ ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=True) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=True)
		Jet["HT"] = ak.sum(Jet.Pt,axis = 1,keepdims=False) #+ ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=False) + ak.sum(JetDown_HT.PtTotUncDown,axis = 1,keepdims=False)
		Jet["MHT_x"] = ak.sum(Jet_MHT.Pt*np.cos(Jet_MHT.phi),axis=1,keepdims=False) #+ ak.sum(JetUp_MHT.PtTotUncUp*np.cos(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.cos(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet["MHT_y"] = ak.sum(Jet_MHT.Pt*np.sin(Jet_MHT.phi),axis=1,keepdims=False) #+ ak.sum(JetUp_MHT.PtTotUncUp*np.sin(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.sin(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet["MHT"] = np.sqrt(Jet.MHT_x**2 + Jet.MHT_y**2)
		test_HT = ak.sum(Jet.Pt,axis = 1,keepdims=True)
		if (self.trigger_bit == 39):
			if (self.signal):
				zeroNum = 0
				for HT in ak.ravel(Jet.HT):
					if (HT == 0):
						zeroNum+=1
				print("Before cut Events with 0 HT: %d"%zeroNum)
		#HT_num = 0
		#print("Test 1:")
		#print(ak.sum(Jet.Pt,axis = 1,keepdims=False))
		#print("Test 2:")
		#print(ak.sum(Jet.Pt,axis = 1,keepdims=True))
		#print(ak.ravel(Jet.HT))
		#print(Jet.HT)
		#for x in ak.ravel(Jet.HT):
			#print(x)
		#	HT_num += 1
			#if x == 0:
				#print("Anomolous zero (post-cross cleaning)")
		#print("HT Num (Cross Cleaning): %d"%HT_num)
		#print("HT Len: %d"%len(Jet.HT))
		#print("Pt Len: %d"%len(Jet.Pt))
		#print("Cross Cleaning Applied")
		#print("Len HT = %d"%len(Jet.HT))

		trigger_mask = bit_mask([self.trigger_bit])		
		
		#Apply Cuts
		tau = tau[tau.pt > 30] #pT
		CutFlow_Table.fill(1*np.ones([len(ak.ravel(tau.pt))]))
		print("Taus after pt Cut: %d"%len(ak.ravel(tau.pt)))
		tau = tau[tau.eta < 2.3] #eta
		CutFlow_Table.fill(2*np.ones([len(ak.ravel(tau.pt))]))
		print("Taus after eta cut: %d"%len(ak.ravel(tau.pt)))
		
		a,b = ak.unzip(ak.cartesian([tau,tau], axis = 1, nested = True))
		mval = deltaR(a,b) < 0.8 
		tau["dRCut"] = mval
		tau = tau[ak.any(tau.dRCut, axis = 2) == True]
		
		CutFlow_Table.fill(3*np.ones([len(ak.ravel(tau.pt))]))
		print("Taus after dR cut: %d"%len(ak.ravel(tau.pt)))
		
		#Loose isolation
		tau = tau[tau.decay >= 0.5]	
		CutFlow_Table.fill(4*np.ones([len(ak.ravel(tau.pt))]))
		tau = tau[tau.iso >= 0.5]
		CutFlow_Table.fill(5*np.ones([len(ak.ravel(tau.pt))]))
		print("Taus after iso and decay cut: %d"%len(ak.ravel(tau.pt)))
		

		AK8Jet = AK8Jet[(ak.sum(tau.charge,axis=1) == 0)] #Apply charge conservation cut to AK8Jets
		Muon = Muon[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		Electron = Electron[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		Jet = Jet[(ak.sum(tau.charge,axis=1) == 0)]
		Jet_HT = Jet_HT[(ak.sum(tau.charge,axis=1) == 0)]
		#JetUp_HT = JetUp_HT[(ak.sum(tau.charge,axis=1) == 0)]
		#JetDown_HT = JetDown_HT[(ak.sum(tau.charge,axis=1) == 0)]
		Jet_MHT = Jet_MHT[(ak.sum(tau.charge,axis=1) == 0)]
		#JetUp_MHT = JetUp_MHT[(ak.sum(tau.charge,axis=1) == 0)]
		#JetDown_MHT = JetDown_MHT[(ak.sum(tau.charge,axis=1) == 0)]
		tau = tau[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		CutFlow_Table.fill(6*np.ones([len(ak.ravel(tau.pt))]))
		print("Taus after Charge Conservation cut: %d"%len(tau.pt))
		
		AK8Jet = AK8Jet[ak.num(tau) >= 4]
		Electron = Electron[ak.num(tau) >= 4]
		Muon = Muon[ak.num(tau) >= 4]
		Jet_MHT = Jet_MHT[ak.num(tau) >= 4]
		#JetUp_MHT = JetUp_MHT[ak.num(tau) >= 4]
		#JetDown_MHT = JetDown_MHT[ak.num(tau) >= 4]
		Jet_HT = Jet_HT[ak.num(tau) >= 4]
		#JetUp_HT = JetUp_HT[ak.num(tau) >= 4]
		#JetDown_HT = JetDown_HT[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		tau = tau[ak.num(tau) >= 4] #4 tau events
		CutFlow_Table.fill(7*np.ones([len(ak.ravel(tau.pt))]))
		print("Taus after 4 tau cut: %d"%len(tau.pt))
		
		if (self.trigger_bit == 40):
			Pt_PreTrigg_Arr = ak.ravel(tau.pt)	
			AK8Pt_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetPt))
			AK8Pt_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetPt)
			AK8SoftMass_PreTrigg.fill(ak.ravel(AK8Jet.AK8JetDropMass))
			AK8SoftMass_NoTrigg_Arr = ak.ravel(AK8Jet.AK8JetDropMass)
			AK8Pt_all.fill("No Trigger",ak.ravel(AK8Jet.AK8JetPt))
		
		if (self.trigger_bit == 39):
			#Fill Histograms
			Pt_PreTrigg_Arr = ak.ravel(tau.pt)
			HT_Val_PreTrigger = ak.sum(Jet_HT.Pt, axis = 1, keepdims=True) #+ ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=True) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=True)
			#print(len(Jet_MHT) ==)
			#if (len(Jet_MHT) != len(Jet)):
			#	print("Length mismatch")
			#else:
			#	for x,y in zip(Jet.Pt,Jet_MHT[np.abs(Jet_MHT.eta) < 3.0].Pt):
			#		if (len(x) != len(y)):
			#			print("Dimension mismatch")
			#			print("len(x) = %d"%len(x))
			#			print("len(y) = %d"%len(y))
			
			#Jet["MHT_x"] = ak.sum(Jet_MHT.Pt*np.cos(Jet_MHT.phi),axis=1,keepdims=False) + ak.sum(JetUp_MHT.PtTotUncUp*np.cos(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.cos(JetDown_MHT.phi),axis=1,keepdims=False)
			#Jet["MHT_y"] = ak.sum(Jet_MHT.Pt*np.sin(Jet_MHT.phi),axis=1,keepdims=False) + ak.sum(JetUp_MHT.PtTotUncUp*np.sin(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.sin(JetDown_MHT.phi),axis=1,keepdims=False)
			JetMHT = Jet
			JetHT = Jet
			JetMHT["MHT"] = np.sqrt(Jet.MHT_x**2 + Jet.MHT_y**2)
			JetHT["HT"] = ak.sum(Jet_HT.Pt, axis = 1, keepdims=False) #+ ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=False) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=False)
			JetHT = JetHT[JetHT.HT > 0] #Fix issue with zeros in HT
			JetMHT = JetMHT[JetMHT.MHT > 0] #Fix issue with zeros in HT
			if (self.signal):
				zeroNum = 0
				for HT in ak.ravel(Jet.HT):
					if (HT == 0):
						zeroNum+=1
				print("After cut Events with 0 HT: %d"%zeroNum)

			Jet = Jet[Jet.HT > 0] #Remove 0 HT events (??)
			if (self.signal):
				offVal = 0
				for (HT1,HT2) in zip(ak.ravel(Jet.HT),ak.ravel(JetHT.HT)):
					if (HT1 != HT2):
						offVal += 1
				print("Number of disagreeing HT terms = %d"%offVal)
			#if (JetHT.HT != Jet.HT):
			#	print("Huh???")

			#Jet["MHT"] = Jet[Jet.MHT > 0] #Fix issue with zeros in HT
			#print("0 HT: %d"%ak.sum(ak.num(Jet[Jet.HT == 0].HT,axis=0)))

			
			HT_PreTrigg.fill(ak.ravel(HT_Val_PreTrigger[HT_Val_PreTrigger > 0]))
			#HT_NoTrigg_Arr = ak.ravel(HT_Val_PreTrigger[HT_Val_PreTrigger > 0])
			HT_NoTrigg_Arr = ak.ravel(Jet.HT)
			MET_PreTrigg.fill(ak.ravel(Jet.pfMET))
			MET_NoTrigg_Arr = ak.ravel(Jet.pfMET)
			print("MET Len: %d"%len(MET_NoTrigg_Arr))	
			print("HT Len: %d"%len(HT_NoTrigg_Arr))		

		if (self.trigger_cut):
			tau = tau[np.bitwise_and(tau.trigger,trigger_mask) == trigger_mask]
			AK8Jet = AK8Jet[np.bitwise_and(AK8Jet.trigger,trigger_mask) == trigger_mask]
			Jet = Jet[np.bitwise_and(Jet.trigger,trigger_mask) == trigger_mask]

		if (self.offline_cut):
			if (self.trigger_bit == 40 and self.cut_num == 0):
				print("Offline Cut 40")
				print("No Selection: %d"%ak.num(ak.ravel(tau.pt),axis=0))
				tau = tau[ak.all(AK8Jet.AK8JetPt > 400, axis = 1)]
				print("AK8 Jet Pt Selection: %d"%ak.num(ak.ravel(tau.pt),axis=0))
				Jet = Jet[ak.all(AK8Jet.AK8JetPt > 400, axis = 1)]
				AK8Jet = AK8Jet[ak.all(AK8Jet.AK8JetPt > 400, axis = 1)]
				
				tau = tau[ak.all(AK8Jet.AK8JetDropMass > 30, axis = 1)]
				print("Soft Drop Mass Seleciton: %d"%ak.num(ak.ravel(tau.pt),axis=0))
				Jet = Jet[ak.all(AK8Jet.AK8JetDropMass > 30, axis = 1)]
				AK8Jet = AK8Jet[ak.all(AK8Jet.AK8JetDropMass > 30, axis = 1)]
				print("Trigger Cuts applied to all")
			
			if (self.trigger_bit == 40 and self.cut_num == 1):
				print("Applying PF400 Cut Only")	
				tau = tau[ak.all(AK8Jet.AK8JetPt > 400, axis = 1)]
				Jet = Jet[ak.all(AK8Jet.AK8JetPt > 400, axis = 1)]
				AK8Jet = AK8Jet[ak.all(AK8Jet.AK8JetPt > 400, axis = 1)]
			
			if (self.trigger_bit == 40 and self.cut_num == 2):
				print("Applying Trim Mass Cut Only")	
				tau = tau[ak.all(AK8Jet.AK8JetDropMass > 30, axis = 1)]
				Jet = Jet[ak.all(AK8Jet.AK8JetDropMass > 30, axis = 1)]
				AK8Jet = AK8Jet[ak.all(AK8Jet.AK8JetDropMass > 30, axis = 1)]
			
			if (self.trigger_bit == 39):
				print("Offline Cut 39")
				print("No Selection: %d"%ak.num(ak.ravel(tau.pt),axis=0))
				
				tau = tau[ak.all(Jet.HT > 500, axis = 1)]
				print("HT Selection: %d"%ak.num(ak.ravel(tau.pt),axis=0))
				AK8Jet = AK8Jet[ak.all(Jet.HT > 500, axis = 1)]
				Jet = Jet[ak.all(Jet.HT > 500, axis = 1)]
				#JetMHT = JetMHT[ak.all(JetHT.HT > 500, axis = 1)]
				
				tau = tau[ak.all(Jet.pfMET > 100, axis = 1)]
				print("pfMet Selection: %d"%ak.num(ak.ravel(tau.pt),axis=0))
				AK8Jet = AK8Jet[ak.all(Jet.pfMET > 100, axis = 1)]
				#JetMHT = JetMHT[ak.all(Jet.pfMET > 100, axis = 1)]
				Jet = Jet[ak.all(Jet.pfMET > 100, axis = 1)]

				tau = tau[ak.all(Jet.MHT > 100, axis = 1)]
				print("MHT Selection: %d"%ak.num(ak.ravel(tau.pt),axis=0))
				AK8Jet = AK8Jet[ak.all(Jet.MHT > 100, axis = 1)]
				Jet = Jet[ak.all(Jet.MHT > 100, axis = 1)]
				#JetHT = JetHT[ak.all(JetMHT.MHT > 100, axis = 1)]
				#JetMHT = JetMHT[ak.all(Jet.MHT > 100, axis = 1)]
				
				tau = tau[ak.all(Jet.PFLooseId, axis = 1)]
				print("Loose ID: %d"%ak.num(ak.ravel(tau.pt),axis=0))
				AK8Jet = AK8Jet[ak.all(Jet.PFLooseId, axis = 1)]
				#JetHT = JetHT[ak.all(Jet.PFLooseId, axis = 1)]
				#JetMHT = JetMHT[ak.all(Jet.PFLooseId, axis = 1)]
				Jet = Jet[ak.all(Jet.PFLooseId, axis = 1)]

				print("Trigger Cuts applied to all")

			
		tau_plus = tau[tau.charge > 0]
		tau_minus = tau[tau.charge < 0]

		#Construct all possible valid ditau pairs
		tau_plus1, tau_plus2 = ak.unzip(ak.combinations(tau_plus,2))
		tau_minus1, tau_minus2 = ak.unzip(ak.combinations(tau_minus,2))
		print(tau_plus1.eta)
		
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
			#pre_triggernum = ak.num(AK8Pt_NoTrigg_Arr,axis=0)
			pre_triggernum = ak.num(Pt_PreTrigg_Arr,axis=0)
			print("Number = %d"%pre_triggernum)
			Pt_PostTrigg_Arr = ak.ravel(tau.pt)
			#post_triggernum = ak.num(AK8Pt_Trigg_Arr,axis=0)
			post_triggernum = ak.num(Pt_PostTrigg_Arr,axis=0)
			print("Number = %d"%post_triggernum)
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
			#pre_triggernum = ak.num(MET_NoTrigg_Arr,axis=0)
			pre_triggernum = ak.num(Pt_PreTrigg_Arr,axis=0)
			print("Number = %d"%pre_triggernum)
			Pt_PostTrigg_Arr = ak.ravel(tau.pt)
			#post_triggernum = ak.num(MET_Trigg_Arr,axis=0)	
			post_triggernum = ak.num(Pt_PostTrigg_Arr,axis=0)
			print("Number = %d"%post_triggernum)
			
			if (self.signal):	
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
					"Cutflow_hist": CutFlow_Table 
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
					"post_trigger_num": post_triggernum,
					"Cutflow_hist": CutFlow_Table 
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

	#filebase = "~/Analysis/BoostedTau/TriggerEff/2018_Samples/GluGluToRadionToHHTo4T_M-"
	signal_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v1_Hadd/GluGluToRadionToHHTo4T_M-"
	table_dict = {}
	debug_dict = {}
	title_dict = {"1000": "1 TeV Signal", "2000": "2 TeV Signal", "3000": "3 TeV Signal", "ZZ4l": r"\(ZZ \rightarrow 4l\)", "top": "Top Background"}
	#file_base = "~/Analysis/BoostedTau/TriggerEff/2018_Background/"
	background_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v2_Hadd/"
	background_dict = {"ZZ4l" : r"$ZZ \rightarrow 4l$", "top": "Top Background"}
	#file_dict = {"ZZ4l": [background_base + "ZZ4l.root"], "top": [background_base + "Tbar-tchan.root",background_base + "Tbar-tW.root",background_base + "T-tchan.root"]}
	#file_dict = {"top": [background_base + "Tbar-tchan.root",background_base + "Tbar-tW.root",background_base + "T-tchan.root"]}
	#file_dict = {"top": [background_base + "Tbar-tW.root",background_base + "T-tchan.root"]}
	file_dict = {"top": [background_base + "TTTo2L2Nu.root",background_base + "TTToSemiLeptonic.root",background_base + "TTToHadronic.root"]}
	#file_dict = {"top": [background_base + "Tbar-tchan.root",background_base + "T-tW.root"]}
	
	for i in range(3):
		if (i == 0):
			#continue
			use_trigger = True
			use_offline= False
			table_title = "Online Trigger Efficiency Table"
			table_file_name = "Trigger_NoCuts"
		if (i == 1):
			use_trigger = False
			use_offline = True
			table_title = "Offline Cuts Efficiency Table"
			table_file_name = "NoTrigger_Cuts"
			#break
		if (i == 2):
			#continue
			use_trigger = True
			use_offline = True
			table_title = "Online Trigger and Offline Cuts Efficiency Table"
			table_file_name = "Trigger_Cuts"
		#if (i == 3):
		#	use_trigger = False 
		#	use_offline = True
		#	table_title = "Trigger and Offline Cuts Efficiency Table"
		#	table_file_name = "Trigger_Cuts"
	
		#Signal
		for mass_str in mass_str_arr:
			fileName = signal_base + mass_str + ".root"
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
				p2 = TriggerStudies(trigger_bit, trigger_cut = use_trigger, offline_cut = use_offline, signal = True)
				trigger_out = p2.process(events)
				
				if (trigger_bit == 39):
					mass_eff_arr[0] = trigger_out["boosted_tau"]["post_trigger_num"]/trigger_out["boosted_tau"]["pre_trigger_num"]
				if (trigger_bit == 40):
					mass_eff_arr[1] = trigger_out["boosted_tau"]["post_trigger_num"]/trigger_out["boosted_tau"]["pre_trigger_num"]
			
				table_dict[title_dict[mass_str]] = mass_eff_arr #Update dictionary
			
				#Produce cutflow table
				if (i == 0):
					fig, ax = plt.subplots()
					trigger_out["boosted_tau"]["Cutflow_hist"].plot1d(ax=ax)
					#ax.set_yscale('log')
					plt.ylim(bottom = 1)
					plt.title("Cutflow Table Mass: " + mass_str[0] + "." + mass_str[1] + " TeV")
					plt.savefig("CutFlowTable_" + mass_str)
					plt.close()

			#Debugging trigger 40
			new_eff_arr = [-1,-1]
			debug_1 = TriggerStudies(40, trigger_cut = use_trigger, offline_cut = use_offline, signal = True, cut_num = 1)
			out_1 = debug_1.process(events)
			new_eff_arr[0] = out_1["boosted_tau"]["post_trigger_num"]/out_1["boosted_tau"]["pre_trigger_num"]
			debug_2 = TriggerStudies(40, trigger_cut = use_trigger, offline_cut = use_offline, signal = True, cut_num = 2)
			out_2 = debug_2.process(events)
			new_eff_arr[1] = out_2["boosted_tau"]["post_trigger_num"]/out_2["boosted_tau"]["pre_trigger_num"]
		
			debug_dict[title_dict[mass_str]] = new_eff_arr #Update debugging dictionary 


		iterative_runner = processor.Runner(
			executor = processor.IterativeExecutor(compression=None),
			schema=BaseSchema
		)
		#Background 
		for background_name, title in background_dict.items():
			if (background_name == "ZZ4l"):
				events = NanoEventsFactory.from_root(
					background_base + background_name + ".root",
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
					#trigger_out = iterative_runner(file_dict, treename="mutau_tree",processor_instance=TriggerStudies(trigger_bit, trigger_cut = use_trigger, offline_cut = use_offline, signal = False)) 
					trigger_out = iterative_runner(file_dict, treename="4tau_tree",processor_instance=TriggerStudies(trigger_bit, trigger_cut = use_trigger, offline_cut = use_offline, signal = False)) 
			
				if (trigger_bit == 40):
					trigger_hist_dict_1d = trigger_AK8Jet_hist_dict_1d 
					trigger_hist_dict_2d = trigger_AK8Jet_hist_dict_2d 
			
				if (trigger_bit == 39):
					trigger_hist_dict_1d = trigger_MTHTJet_hist_dict_1d  
					trigger_hist_dict_2d = trigger_MTHTJet_hist_dict_2d 
			
				for var_name, hist_name_arr in trigger_hist_dict_1d.items():
					if (background_name == "ZZ4l"):
						print("Efficiency = %f"%(trigger_out["boosted_tau"]["post_trigger_num"]/trigger_out["boosted_tau"]["pre_trigger_num"]))
						if (trigger_bit == 39):
							eff_arr[0] = trigger_out["boosted_tau"]["post_trigger_num"]/trigger_out["boosted_tau"]["pre_trigger_num"]	
						if (trigger_bit == 40):
							eff_arr[1] = trigger_out["boosted_tau"]["post_trigger_num"]/trigger_out["boosted_tau"]["pre_trigger_num"]	
					else:
						#trigger_out[background_name]["boosted_tau"][var_name].plot1d(ax=ax)
						#trigger_out[background_name][var_name].plot1d(ax=ax)
						print("Efficiency = %f"%(trigger_out[background_name]["post_trigger_num"]/trigger_out[background_name]["pre_trigger_num"]))
						if (trigger_bit == 39):
							eff_arr[0] = trigger_out[background_name]["post_trigger_num"]/trigger_out[background_name]["pre_trigger_num"]	
						if (trigger_bit == 40):
							eff_arr[1] = trigger_out[background_name]["post_trigger_num"]/trigger_out[background_name]["pre_trigger_num"]	
	
		
			table_dict[title_dict[background_name]] = eff_arr
			
			#Produce cutflow table
			if (i == 0):
				fig, ax = plt.subplots()
				if (background_name == "ZZ4l"):
					trigger_out["boosted_tau"]["Cutflow_hist"].plot1d(ax=ax)
					plt.title(r"Cutflow Table $ZZ \rightarrow 4l$ Background")
				else:
					trigger_out[background_name]["Cutflow_hist"].plot1d(ax=ax)
					plt.title("Cutflow Table Top background")
				#ax.set_yscale('log')
				plt.ylim(bottom = 1)
				plt.savefig("CutFlowTable_" + background_name)
				plt.close()
		
		#Set up debugging table
			new_eff_arr = [-1,-1]
			if (background_name == "ZZ4l"):
				debug_1 = TriggerStudies(40, trigger_cut = use_trigger, offline_cut = use_offline, signal = False, cut_num = 1)
				out_1 = debug_1.process(events)
				new_eff_arr[0] = out_1["boosted_tau"]["post_trigger_num"]/trigger_out["boosted_tau"]["pre_trigger_num"]
				debug_2 = TriggerStudies(40, trigger_cut = use_trigger, offline_cut = use_offline, signal = False, cut_num = 2)
				out_2 = debug_2.process(events)
				new_eff_arr[1] = out_2["boosted_tau"]["post_trigger_num"]/trigger_out["boosted_tau"]["pre_trigger_num"]
				debug_dict[title_dict[background_name]] = new_eff_arr	
			else:
				out_1 = iterative_runner(file_dict, treename="4tau_tree",processor_instance=TriggerStudies(trigger_bit, trigger_cut = use_trigger, offline_cut = use_offline, signal = False, cut_num = 1)) 
				new_eff_arr[0] = out_1[background_name]["post_trigger_num"]/trigger_out[background_name]["pre_trigger_num"]
				out_2 = iterative_runner(file_dict, treename="4tau_tree",processor_instance=TriggerStudies(trigger_bit, trigger_cut = use_trigger, offline_cut = use_offline, signal = False, cut_num = 2)) 
				new_eff_arr[1] = out_2[background_name]["post_trigger_num"]/trigger_out[background_name]["pre_trigger_num"]
				debug_dict[title_dict[background_name]] = new_eff_arr	

		#Set up efficiency table
		DrawTable(table_title,table_file_name,table_dict)
		TriggerDebuggTable("Debug Table",debug_dict)	

