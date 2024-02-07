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

class PlottingObj(processor.ProcessorABC):
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
				"eta": events.jetEta,
				"phi": events.jetPhi,
				"trigger": events.HLTJet,
			},
			with_name="PFJetArray",
			behavior=candidate.behavior,
		)
		
		#Histograms
		#HT_Dist = hist.Hist.new.Reg(40,0, 4000., label = "HT (GeV)").Double()
		#MET_Dist = hist.Hist.new.Reg(30,0,1200., label = "MET (GeV)").Double()
		#MHT_Dist = hist.Hist.new.Reg(30,0,1200., label = "MHT (GeV)").Double()
		#AK8PT_Dist = hist.Hist.new.Reg(40,0,400.,label = r"AK8 Jet $p_T$ (GeV)").Double()
		
		#MHT
		Jet_MHT = Jet[Jet.Pt > 30]
		Jet_MHT = Jet_MHT[np.abs(Jet_MHT.eta) < 5]
		Jet_MHT = Jet_MHT[Jet_MHT.PFLooseId > 0.5]
		Jet_MHT["MHT_x"] = ak.sum(Jet_MHT.Pt*np.cos(Jet_MHT.phi),axis=1,keepdims=False) #+ ak.sum(JetUp_MHT.PtTotUncUp*np.cos(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.cos(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet_MHT["MHT_y"] = ak.sum(Jet_MHT.Pt*np.sin(Jet_MHT.phi),axis=1,keepdims=False) #+ ak.sum(JetUp_MHT.PtTotUncUp*np.sin(JetUp_MHT.phi),axis=1,keepdims=False) + ak.sum(JetDown_MHT.PtTotUncDown*np.sin(JetDown_MHT.phi),axis=1,keepdims=False)
		Jet_MHT["MHT"] = np.sqrt(Jet_MHT.MHT_x**2 + Jet_MHT.MHT_y**2)
		#MHT_NoCrossCleaning.fill(ak.ravel(Jet_MHT.MHT))
		#MHT_Acc_NoCrossingClean = hist.accumulators.Mean().fill(ak.ravel(Jet_MHT.MHT))
		print("Jet MHT Defined:")
		
		#HT Seleciton (new)
		#tau_temp1,HT = ak.unzip(ak.cartesian([tau,Jet_MHT], axis = 1, nested = True))
		Jet_HT = Jet[Jet.Pt > 30]
		Jet_HT = Jet_HT[np.abs(Jet_HT.eta) < 3]
		Jet_HT = Jet_HT[Jet_HT.PFLooseId > 0.5]
		HT,tau_temp1 = ak.unzip(ak.cartesian([Jet_HT,tau], axis = 1, nested = True))
		
		#Get Cross clean free histograms
		#HT_Var_NoCrossClean = ak.sum(Jet.Pt,axis = 1,keepdims=True) #+ ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=True) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=True)
		#MET_NoCrossCleaning.fill(ak.ravel(Jet.pfMET))
		#MET_Acc_NoCrossClean = hist.accumulators.Mean().fill(ak.ravel(Jet.pfMET))
		#HT_NoCrossCleaning.fill(ak.ravel(HT_Var_NoCrossClean))
		#HT_num = 0
		#for x in ak.sum(Jet.Pt,axis = 1,keepdims=False):
		#	HT_num += 1
			
		#print(ak.sum(Jet.Pt,axis = 1,keepdims=True))
		#print("HT Num (No Cross Cleaning): %d"%HT_num)
		#HT_Acc_NoCrossClean = hist.accumulators.Mean().fill(ak.ravel(HT_Var_NoCrossClean[HT_Var_NoCrossClean > 0]))
		#HT_Acc_NoCrossClean = hist.accumulators.Mean().fill(ak.ravel(HT_Var_NoCrossClean))
	
		mval_temp = deltaR(tau_temp1,HT) >= 0.5
		#print(mval_temp)
		if (len(Jet.Pt) != len(mval_temp)):
			print("Things aren't good")
			if (len(Jet.Pt) > len(mval_temp)):
				print("More Jets than entries in mval_temp")
			if (len(Jet.Pt) < len(mval_temp)):
				print("Fewer entries in Jets than mval_temp")

		Jet_HT["dR"] = mval_temp
		HT_Val_NoCuts = ak.sum(Jet_HT.Pt,axis = 1,keepdims=True) #+ ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=True) + ak.sum(JetDown_HT.PtTotUncDown,axis=1,keepdims=True)
		Jet["HT"] = ak.sum(Jet_HT.Pt,axis = 1,keepdims=False) #+ ak.sum(JetUp_HT.PtTotUncUp,axis = 1,keepdims=False) + ak.sum(JetDown_HT.PtTotUncDown,axis = 1,keepdims=False)
		Jet["MHT_x"] = ak.sum(Jet_MHT.Pt*np.cos(Jet_MHT.phi),axis=1,keepdims=False)
		Jet["MHT_y"] = ak.sum(Jet_MHT.Pt*np.sin(Jet_MHT.phi),axis=1,keepdims=False)
		Jet["MHT"] = np.sqrt(Jet.MHT_x**2 + Jet.MHT_y**2)
		#Jet["MHT"] = Jet_MHT.MHT

		#Tau Selections
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
		Jet = Jet[(ak.sum(tau.charge,axis=1) == 0)]
		Jet_HT = Jet_HT[(ak.sum(tau.charge,axis=1) == 0)]
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
		Jet_MHT = Jet_MHT[ak.num(tau) >= 4]
		Jet_HT = Jet_HT[ak.num(tau) >= 4]
		Jet = Jet[ak.num(tau) >= 4]
		tau = tau[ak.num(tau) >= 4] #4 tau events
	
		
		#Fill Histograms
		#AK8PT_Dist.fill(ak.ravel(AK8Jet.AK8JetPt))
		#MHT_Dist.fill(ak.ravel(Jet.MHT))
		#HT_Dist.fill(ak.ravel(Jet.HT))
		#MET_Dist.fill(ak.ravel(Jet.pfMET))
		
		#Output arrays
		AK8PT_Arr = ak.ravel(AK8Jet.AK8JetPt)
		counts = ak.num(Jet.HT)
		counts = counts[counts != 0]
		MHT_Arr = ak.ravel(Jet.MHT)
		MHT_Arr = ak.unflatten(MHT_Arr,counts)
		MHT_Arr = MHT_Arr[:,0]
		HT_Arr = ak.ravel(Jet.HT)
		HT_Arr = ak.unflatten(HT_Arr,counts)
		HT_Arr = HT_Arr[:,0]
		MET_Arr = ak.ravel(Jet.pfMET)
		MET_Arr = ak.unflatten(MET_Arr,counts)
		MET_Arr = MET_Arr[:,0]

		return{
			dataset: {
				"HT_Arr": HT_Arr,
				"MET_Arr": MET_Arr,
				"MHT_Arr": MHT_Arr,
				"AK8PT_Arr": AK8PT_Arr
			}
		}	

	
	def postprocess(self, accumulator):
		pass	

if __name__ == "__main__":
	mass_str_arr = ["1000","3000"]
	mass_legend = {"1000": r"$m_\phi$ = 1 TeV", "3000": r"$m_\phi$ = 3 TeV"}
	signal_base = "root://cmseos.fnal.gov//store/user/abdollah/SkimBoostedHH4t/2018/4t/v1_Hadd/GluGluToRadionToHHTo4T_M-"
	#print("Test")

	file_dict = {r"$m_\phi$ = 1 TeV": [signal_base + "1000.root"], r"$m_\phi$ = 3 TeV": [signal_base + "3000.root"]}	
	Array_list = ["HT_Arr", "MET_Arr", "MHT_Arr", "AK8PT_Arr"]
	figure_dict = {"HT_Arr": "Jet_HT_Dist", "MET_Arr":"Jet_MET_Dist", "MHT_Arr": "Jet_MHT_Dist","AK8PT_Arr": "AK8Jet_PT_Dist"}
	title_dict = {"HT_Arr": "Jet HT", "MET_Arr":"pfMET", "MHT_Arr": "Jet MHT","AK8PT_Arr": r"AK8Jet $p_T$"}
	hist_dict = {"HT_Arr": hist.Hist.new.StrCat([r"$m_\phi$ = 1 TeV",r"$m_\phi$ = 3 TeV"]).Reg(40,0, 4000., label = "HT (GeV)").Double(), 
				"MET_Arr": hist.Hist.new.StrCat([r"$m_\phi$ = 1 TeV",r"$m_\phi$ = 3 TeV"]).Reg(30,0,1200., label = "MET (GeV)").Double(),
				"MHT_Arr":hist.Hist.new.StrCat([r"$m_\phi$ = 1 TeV",r"$m_\phi$ = 3 TeV"]).Reg(30,0,1200., label = "MHT (GeV)").Double(),
				"AK8PT_Arr": hist.Hist.new.StrCat([r"$m_\phi$ = 1 TeV",r"$m_\phi$ = 3 TeV"]).Reg(40,0,400.,label = r"AK8 Jet $p_T$ (GeV)").Double() 
				}

	iterative_runner = processor.Runner(
		executor = processor.IterativeExecutor(compression=None),
		schema=BaseSchema
	)
	runner = iterative_runner(file_dict, treename="4tau_tree",processor_instance=PlottingObj())
	for Arr_name in Array_list:
		fig,ax = plt.subplots()
		for mass in mass_str_arr:
			#if (Arr_Name != "AK8PT_Arr"):		
			#	hist_dict[Arr_name].fill(mass_legend[mass],ak.runner[mass_legend[mass]][Arr_name])
			#else:
			hist_dict[Arr_name].fill(mass_legend[mass],runner[mass_legend[mass]][Arr_name])
		hist_dict[Arr_name].plot1d(ax=ax)
		plt.title("CMS Preliminary",loc="left")
		plt.title(title_dict[Arr_name])
		ax.legend(title="Legend")
		plt.savefig(figure_dict[Arr_name])
		plt.close()
		


