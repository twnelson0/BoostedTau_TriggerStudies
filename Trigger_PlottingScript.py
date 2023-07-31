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

def doublet_gen(events, builder): #This needs to be completely reworked, no loops!!
	for tau in events: 
		builder.begin_list()
		ntau = len(tau)
		#print(ntau)
		#Set up set of tau indicies
		temp_arr = []
		for i in range(ntau):
			temp_arr.append(i)
		indx_set = set(temp_arr)
	
		test_pairs = []
		for i in range(1,ntau):
			if (tau[0].charge + tau[i].charge == 0):
				test_pairs.append([0,i])
		#Obtain tau pairings based on minimum Delta R
		first_pair = test_pairs[0]
		second_pair = []
		min_dR = -1
		for pair in test_pairs:
			if (pair == test_pairs[0]):
				min_dR = deltaR(tau[pair[0]],tau[pair[1]])
			else:
				if (deltaR(tau[pair[0]],tau[pair[1]]) < min_dR):
					min_dR = deltaR(tau[pair[0]],tau[pair[1]])
					first_pair = pair
		
		#Obtain other pairirng
		for indx in first_pair:
			indx_set.remove(indx)
		for j in indx_set:
			second_pair.append(j)
		
		#Store taus ordered by pairing
		builder.begin_tuple(4)
		builder.index(0).integer(first_pair[0])
		builder.index(1).integer(first_pair[1])
		builder.index(2).integer(second_pair[0])
		builder.index(3).integer(second_pair[1])
		builder.end_tuple()		
		builder.end_list()
	
	return builder	

def min_dR(events):
	outArr = []
	for tau in events:
		ntau = len(tau)
		
		#Set up set of tau indicies
		temp_arr = []
		for i in range(ntau):
			temp_arr.append(i)
		indx_set = set(temp_arr)
	
		test_pairs = []
		for i in range(1,ntau):
			if (tau[0].charge + tau[i].charge == 0):
				test_pairs.append([0,i])
		#Obtain tau pairings based on minimum Delta R
		first_pair = test_pairs[0]
		min_dR = -1
		for pair in test_pairs:
			if (pair == test_pairs[0]):
				min_dR = deltaR(tau[pair[0]],tau[pair[1]])
			else:
				if (deltaR(tau[pair[0]],tau[pair[1]]) < min_dR):
					min_dR = deltaR(tau[pair[0]],tau[pair[1]])
		
		#print(min_dR)	
		outArr.append(min_dR)
		#print(len(aout_Arr))
	return outArr	

def iso_fun(tau, iso_arr, iso_lim = 0.4):
	#iso_arr = np.array([])
	iso_arr.begin_list()
	for evnt in range(len(tau.eta)):
		#evnt_arr = np.array([])
		N = len(tau.eta[evnt]) 
		if (N%2 == 1):
			tup_size = N*(N-1)/2
		else:
			tup_size = (N-1)*N/2
		iso_arr.begin_tuple(tup_size)
		tpl_idx = 0
		for i in range(len(tau.eta[evnt])):
			for j in range(i + 1, len(tau.eta[evnt])):
				#evnt_arr = np.append(evnt_arr,(deltaR(tau[evnt],tau[evnt]) < iso_lim))
				#iso_arr.boolean((deltaR(tau[evnt][i],tau[evnt][j]) < iso_lim))
				iso_arr.index(tpl_idx).boolean((deltaR(tau[evnt][i],tau[evnt][j]) < iso_lim))
				tpl_idx += 1
		#iso_arr = np.append(iso_arr,evnt_arr)
		iso_arr.end_tuple()
	
	iso_arr.end_list()

	return iso_arr


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
				"iso": events.boostedTauByVLooseIsolationMVArun2v1DBoldDMwLTNew,
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
			.Reg(50,0,500., name = "mass1", label=r"$m_{\tau \tau} [GeV]$")
			.Int64()
		)
		ditau_mass2_hist = (
			hist.Hist.new
			.Reg(50,0,500., name = "mass2", label=r"$m_{\tau \tau} [GeV]$")
			.Int64()
		)
		dimass_all_hist = (
			hist.Hist.new
			.StrCat(["Pair 1","Pair 2"], name = "ditau_mass")
            .Reg(50, 0, 600., name="ditau_mass_all", label=r"$m_{\tau\tau}$ [GeV]") 
            .Int64()
		)
			
		#fourtau_cut = (ak.num(tau)==4) & (ak.all(tau.iso, axis=1)==True) & (ak.sum(tau.charge,axis=1) == 0) #4 tau, charge and isolation cut (old Cut)
		print("Isolation Stuff")
		#Apply cuts
		isoTau = tau[tau.iso]
		isoTau = isoTau[(ak.sum(isoTau.charge,axis=1) == 0)]
		print(ak.num(isoTau) == 4) #This is not doing what I think it should be doing, I appear to be losing > 100 events when I should only be loosing 2
		print(len(isoTau.nBoostedTau))
		fourTau = isoTau[ak.num(isoTau) == 4]
		print(len(fourTau.nBoostedTau))
		#print(len(isoTau))
		#print(len([(len(x.pt) == 4) for x in isoTau]))
		#fourTau = isoTau[(len(x.pt) == 4) for x in isoTau]
		drp_evnt = 0
		for x in isoTau:
			if (ak.num(x,axis=0) > 4):
				print(ak.num(x,axis=0))
				print(x.pt)
				drp_evnt+=1
		print(len(isoTau) - drp_evnt)

		#print(len(tau[iso_cut].nBoostedTau))
		fourtau_cut = (ak.num(tau)==4) & (ak.all(tau.iso, axis=1)==True) & (ak.sum(tau.charge,axis=1) == 0) #4 tau, charge and isolation cut
		#print(fourtau_cut)

		#iso_bool = ak.Array(iso_fun(tau, ak.ArrayBuilder()))
		#print(len(iso_bool))
		#for x in iso_bool:
		#	print(x) 
		#print(len(fourtau_cut) == len(tau))
		#print(len(tau.eta))
		min_drArr = min_dR(tau[fourtau_cut])
		min_drArr = ak.Array([(x < 1) for x in min_drArr])
		leading_tau = tau[fourtau_cut][:,0]
		#print(len(min_drArr) == len(leading_tau))
		#print(min_drArr)
		#print(fourtau_cut)
		#leading_tau = (tau[fourtau_cut])[min_drArr][:,0]
		#subleading_tau = (tau[fourtau_cut])[min_drArr][:,1]
		#thirdleading_tau = (tau[fourtau_cut])[min_drArr][:,2]
		#fourthleading_tau = (tau[fourtau_cut])[min_drArr][:,3]
		
		leading_tau = fourTau[min_drArr][:,0]
		subleading_tau = fourTau[min_drArr][:,1]
		thirdleading_tau = fourTau[min_drArr][:,2]
		fourthleading_tau = fourTau[min_drArr][:,3]
		
		ditau_evnt = doublet_gen(fourTau[min_drArr], ak.ArrayBuilder()).snapshot()
		#print(ditau_evnt)
		#print(ditau_evnt[0])
		#print(ditau_evnt[0][0]["1"])
		#ditau_evnt = [tau[ditau_evnt[i]] for i in range(4)]
		#print(ditau_evnt)
		
		#Fill plots
		pt_hist.fill(leading_tau.pt)
		eta_hist.fill(leading_tau.eta)
		phi_hist.fill(leading_tau.phi)
		pt_all_hist.fill("Leading",leading_tau.pt)
		pt_all_hist.fill("Subleading",subleading_tau.pt)
		pt_all_hist.fill("Third-leading",thirdleading_tau.pt)
		pt_all_hist.fill("Fourth-leading",fourthleading_tau.pt)
		pt4_hist.fill(fourthleading_tau.pt)
		ditau_mass1_hist.fill(mass(tau[fourtau_cut][:,ditau_evnt[0][0]["0"]], tau[fourtau_cut][:,ditau_evnt[0][0]["1"]]))
		dimass_all_hist.fill("Pair 1", mass(tau[fourtau_cut][:,ditau_evnt[0][0]["0"]], tau[fourtau_cut][:,ditau_evnt[0][0]["1"]]))
		ditau_mass2_hist.fill(mass(tau[fourtau_cut][:,ditau_evnt[0][0]["2"]], tau[fourtau_cut][:,ditau_evnt[0][0]["3"]]))
		dimass_all_hist.fill("Pair 2", mass(tau[fourtau_cut][:,ditau_evnt[0][0]["2"]], tau[fourtau_cut][:,ditau_evnt[0][0]["3"]]))
		#ditau_mass1_hist.fill((tau[fourtau_cut][:,ditau_evnt[0][0]["0"]] + tau[fourtau_cut][:,ditau_evnt[0][0]["1"]]).mass)
		#ditau_mass1_hist.fill(tau[fourtau_cut][:,ditau_evnt[0]].mass + tau[fourtau_cut][:,ditau_evnt[1]].mass)

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


#class DoubletPlotting(processor.ProcessorABC):
#	def __init__(self):
#		pass
#	
#	def process(self, events):
#		dataset = events.metadata['dataset']
#		tau_doublet = ak.zip(
#			{
#				"mass": , 
#			}
#		)
#
#		return{
#			dataset: {
#				"entries" : len(events),
#			}
#		}	
#	
#	def postprocess(self, accumulator):
#		pass	

if __name__ == "__main__":
	mass_str_arr = ["1000","2000","3000"]
	#fileName = "GluGluToRadionToHHTo4T_M-1000.root"
	filebase = "GluGluToRadionToHHTo4T_M-"
	
	for mass_str in mass_str_arr:
		fileName = filebase + mass_str + ".root"
		events = NanoEventsFactory.from_root(
			fileName,
			treepath="/4tau_tree",
			schemaclass = BaseSchema,
			metadata={"dataset": "boosted_tau"},
		).events()
		#print(events.boostedTauPt)
		#print(len(events.boostedTauPt))
		print("Any test:")
		print(ak.all(events.boostedTauPt))
		print((ak.all(events.boostedTauPt, axis = 1) <= 20))
		#print(events.boostedTauByVLooseIsolationMVArun2v1DBoldDMwLTNew)
		#print(ak.all(events.boostedTauByVLooseIsolationMVArun2v1DBoldDMwLTNew, axis = 1) == True)
		print(events.boostedTauPt)
		
		#print(len(events.boostedTauPt) == len(ak.any(events.boostedTauPt, axis=1) > 20))
		#for x in events.boostedTauByVLooseIsolationMVArun2v1DBoldDMwLTNew:
		#	if (len(x) == 4):
		#		print(x)		
		
		#for x in events.boostedTauCharge:
		#	if (len(x) == 4 and sum(x) != 0):
		#		print(x)		
		
		#print(events.boostedTauEta[1])
		#print(events.boostedTauPt[1])
		#print(events.boostedTauPt[:, events.leadtauIndex][0])
		#cut = (ak.num(events.boostedTauPt) == 4)
		#print(cut)
		#for x in events.boostedTauPt:
		#	if (x[0] != max(x)):
		#		print(x)
		#for x in events.boostedTauEta:
		#	if (len(x) == 3):
		#		print(x)
#		print(events.leadtauIndex)
#		print(events.nBoostedTau)
		#print(events.boostedTauPt[:, events.leadtauIndex]) #This deosn't work the way I exepcted it to
		p = TauPlotting()
		out = p.process(events)
		break
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
		plt.title(r"Di-tau pair 1 mass")
		plt.savefig("Ditau_Mass1-" + mass_str)
		plt.cla()
		out["boosted_tau"]["mass2"].plot1d(ax=ax)
		plt.title(r"Di-tau pair 2 mass")
		plt.savefig("Ditau_Mass2-" + mass_str)
		plt.cla()
		out["boosted_tau"]["ditau_mass"].plot1d(ax=ax)
		plt.title(r"Di-$\tau$ pair masses")
		ax.legend(title=r"Di-$\tau$ Pair")
		plt.savefig("AllDitauMass_Plot-" + mass_str)

	#print(max(events.leadtauIndex))
	
	#testArr = []
	#for i in range(events.nBoostedTau):
	#	if (i == 0):
	#		indx = 0 + events.leadtauIndex[i]
	#	else:
	#		indx = events.nBoostedTau[i - 1] + events.leadtauIndex[i]
	#	for j in range()
	#for i in range(len(events.nBoostedTau)):
	#	for j in range(events.nBoostedTau[i]):
	#		if (j == events.leadtauIndex[i]):
	#			testArr.append(True)
	#		else:
	#			testArr.append(False)
	#print(testArr) 
		
		
	#pT_plot("GluGluToRadionToHHTo4T_M-1000.root")


