/*******************************************************************************
Main file

Authors: Eric Klinkhammer

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*******************************************************************************/

#include "experimentUtil.h"
#include "AlignmentLearningAgent.cpp"


using std::vector ;
using std::string ;
using namespace Eigen ;

int readTrialNum() {
  int trialNum;
  std:: cout << "Please enter trial number [NOTE: no checks enabled to prevent"
	     << " overwriting existing files, user must make sure trial number "
	     << "is unique]: ";
  std::cin >> trialNum;
  return trialNum;
}

void makeDir(std::string dir) {
  std::string command = "mkdir -p " + dir;
  system(command.c_str());
}

void testDomainOnce(MultiRover* domain, bool output) {
  domain->setVerbose(output);
  domain->InitialiseEpoch();
  domain->ResetEpochEvals();
  //  domain->SimulateEpoch(false);
}

int main() {
  YAML::Node config = YAML::LoadFile("../input/config.yaml");


  
  int trialNum = 1;//readTrialNum();
  string fileDir = "Results/" + std::to_string(trialNum);
  makeDir(fileDir);

  vector< vector<size_t>> inds;
  vector< vector< NeuralNet* > > teams;
  
  MultiRover* domain;
  Objective* o;
  vector< Objective* > objs;
  YAML::Node root = nodeFromYAML(config, "control");
  o = objFromYAML(root, objectiveS);
  objs.push_back(o);
  
  YAML::Node experiments = nodeFromYAML(config, "experiments");
  vector<std::string> experimentStrings;


  YAML::Node controlNode = nodeFromYAML(config, "control");

  vector<std::string> eRewards = {"G", "D"};
  vector<int> eBias = {1, 0};
  vector<int> eSteps = {30, 45};
  vector<int> eCoupling = {3,4};
  vector<double> eWorld = {25.0, 40.0};
  vector<int> ePOI = {6,10};


  // Get all control data
  // for (const auto& eR : eRewards) {
  //   for (const auto& eB : eBias) {
  //     for (const auto& eS : eSteps) {
  // 	for (const auto& eW : eWorld) {
  // 	  for (const auto& eP : ePOI) {
  // 	    for (const auto& eC : eCoupling) {
  // 	      std::cout << "World Size: " << eW << " Bias: " << eB
  // 			<< " Steps: " << eS << " POIs: " << eP
  // 			<< " Reward: " << eR << " Coupling: " << eC << std::endl;
						
  // 	      controlNode[objectiveS][typeS] = eR;
  // 	      controlNode[objectiveS]["coupling"] = eC;
  // 	      controlNode["biasStart"] = eB;
  // 	      controlNode["nSteps"] = eS;
  // 	      controlNode["nPOIs"] = eP;
  // 	      controlNode["world"]["xmax"] = eW;
  // 	      controlNode["world"]["ymax"] = eW;
  // 	      trainingCurves(controlNode, 350, 5, 10);
  // 	    }
  // 	  }
  // 	}
  //     }
  //   }
  // }

  // Load all previous policies
  for (YAML::const_iterator it = experiments.begin(); it != experiments.end(); ++it) {
    std::string experimentName = it->as<std::string>();
    experimentStrings.push_back(experimentName);
  }
  
  VectorXd input;
  for (auto& expKey : experimentStrings) {
    YAML::Node expNode = nodeFromYAML(config, expKey);
    //trainingCurves(expNode, 1000, 5, 10);
    o = objFromYAML(expNode, objectiveS);
    teams.push_back(bestTeamForObjective(expNode, expKey, domain, o, fileDir));
    vector<size_t> ind = fromYAML<vector<size_t>>(expNode, "ind");
    inds.push_back(ind);
    objs.push_back(o);
  } // for each experiment


  // Do all experiments
  root = nodeFromYAML(config, "control");

  vector<double> eLearn = {0.999, 0.99};
  for (const auto& eL : eLearn) {
    for (const auto& eB : eBias) {
      for (const auto& eS : eSteps) {
	for (const auto& eW : eWorld) {
	  for (const auto& eP : ePOI) {
	    for (const auto& eC : eCoupling) {
	      std::cout << "World Size: " << eW << " Bias: " << eB
			<< " Steps: " << eS << " POIs: " << eP
			<< " Learning Rate: " << eL << " Coupling: "
			<< eC << std::endl;
						
	      controlNode["biasStart"] = eB;
	      controlNode["nSteps"] = eS;
	      controlNode["nPOIs"] = eP;
	      controlNode["world"]["xmax"] = eW;
	      controlNode["world"]["ymax"] = eW;

	      G g(eC, 4, 1);
	      o = &g;

	       size_t nRovs  = size_tFromYAML(root, nRovsS);
	       size_t nPOIs  = size_tFromYAML(root, nPOIsS);
	       size_t nSteps = size_tFromYAML(root, nStepsS);
	       
	       string type = stringFromYAML(root, typeS);
	       
	       double xmin = fromYAML<double>(root, xminS);
	       double ymin = fromYAML<double>(root, yminS);
	       double xmax = fromYAML<double>(root, xmaxS);
	       double ymax = fromYAML<double>(root, ymaxS);
	       std::vector<double> world = {xmin, xmax, ymin, ymax};

	       for (size_t eps = 0; eps < 5001; eps += 250) {
		 root[nEpsS] = eps;
    
		 std::vector<double> means;
		 std::vector<double> stddevs;
		 std::vector<double> stderrs;
    
		 for (size_t trial = 0; trial < 5; trial++) {
		   Alignments as(objs, 10, 0.4);
		   as.addAlignments(1000);

		   vector< Agent* > roverTeam;
		   for (size_t i = 0; i < nRovs; i++) {
		     vector<NeuralNet*> netsAgent;
		     for (size_t j = 0; j < teams.size(); j++) {
		       int netI = teams[j].size() > i ? i : 0;
		       netsAgent.push_back(teams[j][netI]);
		     }
		     roverTeam.push_back(new AlignmentLearningAgent(netsAgent, &as, inds, 15, 1.0, eL));
      }
      
		   MultiRover domainD(world, nSteps, nPOIs, nRovs, roverTeam);
      
		   if (eB == 0) {
		     domainD.setBias(false);
		   }
      
		   domainD.InitialiseEpoch();
		   domainD.ResetEpochEvals();
		   domainD.setVerbose(false);
		   trainDomain(&domainD, root, "ABC", fileDir, o);
      
		   std::vector<double> alignScores;
		   vector< size_t > agentsForSim(nRovs, 0);
		   vector< NeuralNet* > trialNets = domainD.getBestNNTeam(o);
		   vector< Agent* > ags = domainD.getAgents();

		   for (size_t a = 0; a < ags.size(); a++) {
		     NeuralNet* agentZ = ags[a]->GetNEPopulation()->GetNNIndex(0);
		     agentZ->SetWeights(trialNets[a]->GetWeightsA(),
					trialNets[a]->GetWeightsB());
		   }
      
		   for (int reps = 0; reps < 100; reps++) {
		     domainD.InitialiseEpoch();
		     alignScores.push_back(domainD.runSim(domainD.createSim(domainD.getNPop()),
							  agentsForSim, o));
		     domainD.ResetEpochEvals();
		   }
		   std::cout << "Alignment scores for trial " << trial << ": "
			     << eps << " " << mean(alignScores) << " "
			     << stddev(alignScores) << " "
			     << statstderr(alignScores) << std::endl;
		   
		   means.push_back(mean(alignScores));
		   stddevs.push_back(stddev(alignScores));
		   stderrs.push_back(statstderr(alignScores));
		 }
		 std::cout << "Trial scores - Eps " << eps << " " << mean(means) << " " << stddev(means) << std::endl;
		 if (eps > 1999) eps += 750;
	       }
	    }
	  }
	}
      }
    }
  }
  
  // std::cout << root << std::endl;
  
  // G g(3,4,1);
  // o = &g;//objFromYAML(root, objectiveS);
  // size_t nRovs  = size_tFromYAML(root, nRovsS);
  // size_t nPOIs  = size_tFromYAML(root, nPOIsS);
  // size_t nSteps = size_tFromYAML(root, nStepsS);
  
  // string type = stringFromYAML(root, typeS);
  
  // double xmin = fromYAML<double>(root, xminS);
  // double ymin = fromYAML<double>(root, yminS);
  // double xmax = fromYAML<double>(root, xmaxS);
  // double ymax = fromYAML<double>(root, ymaxS);
  // std::vector<double> world = {xmin, xmax, ymin, ymax};
  // //trainingCurves(root, 350, 5, 10);
  // //if (true) return 0;

  // for (size_t eps = 4000; eps < 5001; eps += 250) {
  //   root[nEpsS] = eps;
  //   std::cout << "Learning with eps count " << size_tFromYAML(root, nEpsS)
  // 	      << std::endl;
    
  //   std::vector<double> means;
  //   std::vector<double> stddevs;
  //   std::vector<double> stderrs;
    
  //   for (size_t trial = 0; trial < 5; trial++) {
  //     Alignments as(objs, 10, 0.4);
  //     as.addAlignments(1000);

  //     vector< Agent* > roverTeam;
  //     for (size_t i = 0; i < nRovs; i++) {
  // 	vector<NeuralNet*> netsAgent;
  // 	for (size_t j = 0; j < teams.size(); j++) {
  // 	  int netI = teams[j].size() > i ? i : 0;
  // 	  netsAgent.push_back(teams[j][netI]);
  // 	}
  // 	roverTeam.push_back(new AlignmentLearningAgent(netsAgent, &as, inds, 15, 1.0, 0.999));
  //     }
      
  //     MultiRover domainD(world, nSteps, nPOIs, nRovs, roverTeam);
      
  //     int biasStart = intFromYAML(root, biasStartS);
  //     if (biasStart == 0) {
  // 	domainD.setBias(false);
  //     }
      
  //     domainD.InitialiseEpoch();
  //     domainD.ResetEpochEvals();
  //     domainD.setVerbose(false);
  //     trainDomain(&domainD, root, "ABC", fileDir, o);
      
  //     std::vector<double> alignScores;
  //     vector< size_t > agentsForSim(nRovs, 0);
  //     vector< NeuralNet* > trialNets = domainD.getBestNNTeam(o);
  //     vector< Agent* > ags = domainD.getAgents();

  //     for (size_t a = 0; a < ags.size(); a++) {
  // 	NeuralNet* agentZ = ags[a]->GetNEPopulation()->GetNNIndex(0);
  // 	agentZ->SetWeights(trialNets[a]->GetWeightsA(),
  // 			   trialNets[a]->GetWeightsB());
  //     }
      
  //     for (int reps = 0; reps < 100; reps++) {
  // 	domainD.InitialiseEpoch();
  // 	alignScores.push_back(domainD.runSim(domainD.createSim(domainD.getNPop()),
  // 					     agentsForSim, o));
  // 	domainD.ResetEpochEvals();
  //     }
  //     std::cout << "Alignment scores for trial " << trial << ": " << eps << " " << mean(alignScores) << " "
  // 	      << stddev(alignScores) << " " << statstderr(alignScores) << std::endl;
  //     means.push_back(mean(alignScores));
  //     stddevs.push_back(stddev(alignScores));
  //     stderrs.push_back(statstderr(alignScores));
  //   }
  //   std::cout << "Trial scores - Eps " << eps << " " << mean(means) << " " << stddev(means) << std::endl;
  //   if (eps > 1999) eps += 750;
  // }
    //trainDomain(domainD, 1000, true, 20, "", fileDir, "ABC", true, o);
    
  // vector< size_t > agentsForSim(nRovs, 0);
  // for (int m = 1; m < 20; m++) {
  //   //std::cout << "Adding alignment values to map... " << m*50 << std::endl;
  //   as.addAlignments(50);
  //   domainD.setVerbose(false);
  //   std::vector<double> alignScores;

  //   //roverTeam[0]->setOutputBool(true);
  //   for (int reps = 0; reps < 100; reps++) {
  //     //std::cout << "New Epoch..." << std::endl;
  //     domainD.InitialiseEpoch();
  //     alignScores.push_back(domainD.runSim(domainD.createSim(domainD.getNPop()),
  // 					   agentsForSim, o));
  //     domainD.ResetEpochEvals();
  //   }

  //   std::cout << m*50 << " " << mean(alignScores) << " "
  // 	      << stddev(alignScores) << " " << statstderr(alignScores) << std::endl;
  // }
  return 0 ;
}
