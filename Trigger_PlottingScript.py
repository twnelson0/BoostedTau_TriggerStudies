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

class TriggerStudies(processor.ProcessorABC):
	def __init__(self):
		pass
	
	def process(self, events):
		dataset = events.metadata['dataset']
	
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
			.Reg(50, 0, 150., name = "mass1", label=r"$m_{\tau \tau} [GeV]$")
			.Int64()
		)
		ditau_mass2_hist = (
			hist.Hist.new
			.Reg(50, 0, 150., name = "mass2", label=r"$m_{\tau \tau} [GeV]$")
			.Int64()
		)
		dimass_all_hist = (
			hist.Hist.new
			.StrCat(["Pair 1","Pair 2"], name = "ditau_mass")
            .Reg(50, 0, 150., name="ditau_mass_all", label=r"$m_{\tau\tau}$ [GeV]") 
            .Int64()
		)
			
		#Apply cuts/selection
		tau = tau[tau.pt > 30] #pT
		tau = tau[tau.eta < 2.3] #eta
		
		#Loose isolation
		tau = tau[tau.iso1 >= 0.5]
		tau = tau[tau.iso2 >= 0.5]		

		#tau = tau[tau.iso] #Isolation cut
		tau = tau[(ak.sum(tau.charge,axis=1) == 0)] #Charge conservation
		tau = tau[ak.num(tau) == 4] #4 tau events (unsure about this)
		tau_plus = tau[tau.charge > 0]	
		tau_minus = tau[tau.charge < 0]

		#Construct all possible valid ditau pairs
		tau_plus1, tau_plus2 = ak.unzip(ak.combinations(tau_plus,2))
		tau_minus1, tau_minus2 = ak.unzip(ak.combinations(tau_minus,2))

		deltaR11 = deltaR(tau_plus1, tau_minus1)
		deltaR12 = deltaR(tau_plus1, tau_minus2)
		deltaR22 = deltaR(tau_plus2, tau_minus2)
		deltaR21 = deltaR(tau_plus2, tau_minus1)

		pairing_11 = (deltaR11 < deltaR12) & (deltaR11 < deltaR21) & (deltaR11 < deltaR22)
		pairing_12 = (deltaR12 < deltaR11) & (deltaR12 < deltaR21) & (deltaR12 < deltaR22)
		pairing_21 = (deltaR21 < deltaR11) & (deltaR21 < deltaR12) & (deltaR21 < deltaR22)
		pairing_22 = (deltaR22 < deltaR12) & (deltaR22 < deltaR21) & (deltaR22 < deltaR11)
		
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
		dimass_all_hist.fill("Pair 1", ak.ravel(mass(tau_plus1[pairing_11], tau_minus1[pairing_11])))
		dimass_all_hist.fill("Pair 1", ak.ravel(mass(tau_plus2[pairing_22], tau_minus2[pairing_22])))
		dimass_all_hist.fill("Pair 2", ak.ravel(mass(tau_plus1[pairing_12], tau_minus2[pairing_12])))
		dimass_all_hist.fill("Pair 2", ak.ravel(mass(tau_plus2[pairing_21], tau_minus1[pairing_21])))
		
		ditau_mass1_hist.fill(ak.ravel(mass(tau_plus1[pairing_11], tau_minus1[pairing_11])))	
		ditau_mass1_hist.fill(ak.ravel(mass(tau_plus2[pairing_22], tau_minus2[pairing_22])))	
		ditau_mass2_hist.fill(ak.ravel(mass(tau_plus1[pairing_12], tau_minus2[pairing_12])))	
		ditau_mass2_hist.fill(ak.ravel(mass(tau_plus2[pairing_21], tau_minus1[pairing_21])))	

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


