/*
This macro shows how to compute jet energy scale.
root -l examples/Example4.C'("delphes_output.root", "plots.root")'
*/

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "ExRootTreeReader.h"
#include "ExRootResult.h"
#else
class ExRootTreeReader;
class ExRootResult;
#endif

class ExRootResult;
class ExRootTreeReader;

//------------------------------------------------------------------------------

void AnalyseEvents(ExRootTreeReader *treeReader,  const char *outputFile_part)
{
  TClonesArray *branchGenJet = treeReader->UseBranch("GenJet");
  TClonesArray *branchParticle = treeReader->UseBranch("Particle");
  TClonesArray *branchEvent = treeReader->UseBranch("Event");
  TClonesArray *branchJet = treeReader->UseBranch("Jet");
  TClonesArray *branchEFlowTrack = treeReader->UseBranch("EFlowTrack");
  TClonesArray *branchEFlowPhoton = treeReader->UseBranch("EFlowPhoton");
  TClonesArray *branchEFlowNeutralHadron = treeReader->UseBranch("EFlowNeutralHadron");
  
  Long64_t allEntries = treeReader->GetEntries();
  ofstream myfile_det;
  ofstream myfile_part;

  cout << "** Chain contains " << allEntries << " events" << endl;

  Jet  *genjet;
  Jet *jet;
  GenParticle *particle;
  GenParticle *muparticle;
  GenParticle *genparticle;
  GenParticle *motherparticle;
  TObject *object;
  Photon *photon;
  Track *track;
  Tower *tower;
  
  TLorentzVector genJetMomentum;
  TLorentzVector jetMomentum;
  TLorentzVector myMomentum;
  
  TLorentzVector incomingelectron;
  TLorentzVector incomingpositron;
  TLorentzVector outgoingphoton;
  
  Long64_t entry;

  Int_t i, j;

  myfile_part.open (outputFile_part);

  // Loop over all events
  for(entry = 0; entry < allEntries; ++entry)
  {
    //if (entry > 10000) break;
    // Load selected branches with data from specified event
    treeReader->ReadEntry(entry);
    HepMCEvent *event = (HepMCEvent*) branchEvent -> At(0);
    //std::cout << "weight : " << event->Weight << std::endl;
    
    if(entry%500 == 0) cout << "Event number: "<< entry <<endl;

    //myfile_part << event->Weight << " ";
    
    bool passevent = false;
    for(i = 0; i < branchJet->GetEntriesFast(); ++i){
      jet = (Jet*) branchJet->At(i);
      if (jet->PT > 1200 && branchJet->GetEntriesFast() > 1) passevent = true;
    }
    for(i = 0; i < branchJet->GetEntriesFast(); ++i){
      jet = (Jet*) branchJet->At(i);
      jetMomentum = jet->P4();
      if (i > 1 or (passevent == false)) continue;
      //std::cout << " " << i << " " << jet->PT << std::endl;
      myfile_part << entry << " " << i << " J " << jet->PT << " " << jet->Eta << " " << jet->Phi << " " << jetMomentum.M() << " " << jet->BTag << " " << jet->Tau[0] << " " << jet->Tau[1] << " " << jet->Tau[2] << " " << jet->Tau[3] << " " << jet->Tau[4] << " ";

      //std::cout << "jet momentuM " << jet->PT*cos(jet->Phi) << std::endl;
      double yyy = 0.;
      for(j = 0; j < jet->Constituents.GetEntriesFast(); ++j){
	object = jet->Constituents.At(j);
        if(object == 0) continue;
        if(object->IsA() == GenParticle::Class()){
          particle = (GenParticle*) object;
          //std::cout << "           gen part " << particle->PID << " " << particle->PT << std::endl;
        }
        else if(object->IsA() == Track::Class()){
          track = (Track*) object;
	  myfile_part << " P " << track->PT << " " << track->Eta << " " << track->Phi << " ";
          //std::cout << "           track " << track->PT << std::endl;
        }
        else if(object->IsA() == Tower::Class()){
          tower = (Tower*) object;
          //std::cout << "           tower " << tower->ET << std::endl;
	  yyy+=tower->ET*cos(tower->Phi);
	  myfile_part << " P " << tower->ET << " " << tower->Eta << " " << tower->Phi << " ";
        }
      }
      //std::cout << "after 2 " << yyy << std::endl;
      myfile_part << std::endl;
    }

  }
}
//------------------------------------------------------------------------------

void LHCO_process(const char *inputFile, const char *outputFile_part)
{
  gSystem->Load("libDelphes");

  TChain *chain = new TChain("Delphes");
  chain->Add(inputFile);

  ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
  ExRootResult *result = new ExRootResult();

  AnalyseEvents(treeReader,outputFile_part);

  cout << "** Exiting..." << endl;

  delete result;
  delete treeReader;
  delete chain;
}

//------------------------------------------------------------------------------
