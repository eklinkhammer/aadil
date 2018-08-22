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
//#include "AlignmentLearningAgent.cpp"
//#include "AlignmentGuidedAgent.cpp"
#include "Domains/OnePoi.cpp"

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
  MultiRover* domain;
  Objective* o;
  srand(time(NULL));
  std::string fileDir = "filedir";
  
  YAML::Node config = YAML::LoadFile("../input/config.yaml");
  YAML::Node root = nodeFromYAML(config, "control");

  // All experimental setups
  YAML::Node experiments = nodeFromYAML(config, "experiments");
  vector<std::string> experimentStrings;
  for (YAML::const_iterator it = experiments.begin(); it != experiments.end(); ++it) {
    std::string experimentName = it->as<std::string>();
    experimentStrings.push_back(experimentName);
  }
  
  // Experiments that get subpolicies and subobjectives
  // Will read existing policies if saved
  YAML::Node subobjectives = nodeFromYAML(config, "subobjectives");
  vector<std::string> subobjectiveStrings;
  vector< vector<size_t> > inds;
  vector< vector< NeuralNet* > > teams;
  for (YAML::const_iterator it = subobjectives.begin(); it != subobjectives.end(); ++it) {
    std::string objectiveTrainingExpName = it->as<std::string>();
    subobjectiveStrings.push_back(objectiveTrainingExpName);
  }
  
  vector<Objective*> objectives;
  G globalReward(6, 4, 1);
  objectives.push_back(&globalReward);


  // Control rewards
  //G g(3, 4, 1);
  //D d(3, 4, 1);
  
  // Transfer Learning Objectives go in increasing coupling order
  //vector<Objective*> transferLearningObjectives;
  //G transfer1(1, 4, 1);
  //G transfer2(2, 4, 1);
  //G transfer3(3, 4, 1);
  //transferLearningObjectives.push_back(&transfer1);
  //transferLearningObjectives.push_back(&transfer2);
  //transferLearningObjectives.push_back(&transfer3);

  

  // G couple_1(1, 4, 1);
  // G couple_2(2, 4, 1);
  // G couple_3(3, 4, 1);
  // G couple_4(4, 4, 1);
  // std::vector<Objective*> controls = {&couple_1, &couple_2,
  // 				      &couple_3, &couple_4};
  // std::vector<double> rs;
  // for (const auto& obj : controls) {
  //   rs.clear();
  //   domain = getDomain(root);
  //   trainDomain(domain, root, obj->getName(),
  // 		fileDir, obj);
  //   rs = testDomain(domain, obj, 100);
  //   std::cout << "Test for " << obj->getName()
  // 	      << " Mean: " << mean(rs)
  // 	      << " Stddev: " << stddev(rs)
  // 	      << " Stderr: " << statstderr(rs) << std::endl;
  // }
  
  // std::cout << "Completed tests for controls." << std::endl;

  // Get trained neural networks
  for (const auto& objKey : subobjectiveStrings) {
    YAML::Node objExpNode = nodeFromYAML(config, objKey);
    o = objFromYAML(objExpNode, objectiveS);
    teams.push_back(bestTeamForObjective(objExpNode, objKey, domain, o, fileDir));
    vector<size_t> ind = fromYAML<vector<size_t>>(objExpNode, "ind");
    inds.push_back(ind);
    objectives.push_back(o);
  }


  // Set Alignment
  std::cout << "Constructing alignment map.";
  std::flush(std::cout);
  Alignments alignments(objectives, 30, 0.4);
  for (size_t i = 0; i < 30; i++) {
    alignments.addAlignments(250);
    std::cout << '.';
    std::flush(std::cout);
  }

  MultiRover* agentTeamDom;
  for (const auto& expKey : experimentStrings) {
    std::cout << "Experimental configuration: " << expKey << std::endl;

    YAML::Node expNode = nodeFromYAML(config, expKey);
    std::cout << expNode << std::endl;    
    domain = getDomain(expNode);

    // // Test trained neural networks
    vector< Agent* > agents = domain->getAgents();
    vector< size_t > agentsForSim(agents.size(), 0);
    
    for (size_t obj = 0;obj < subobjectiveStrings.size(); obj++) {
      YAML::Node teamNode = nodeFromYAML(config, subobjectiveStrings[obj]);
      agentTeamDom = getDomain(teamNode);
      agents = agentTeamDom->getAgents();
      // this makes sure networks of different sizes have proper inputs
      // Not all agents expect the same size network.
      domain->setTeam(agents);
      
      vector< NeuralNet* > trialNets = teams[obj];
      for (size_t a = 0; a < agents.size(); a++) {
    	NeuralNet* agentZ = agents[a]->GetNEPopulation()->GetNNIndex(0);

    	agentZ->SetWeights(trialNets[a]->GetWeightsA(),
    			   trialNets[a]->GetWeightsB());
      }

      std::vector<double> results;
      for (size_t reps = 0; reps < 100; reps++) {
    	domain->InitialiseEpoch();
    	results.push_back(domain->runSim(domain->createSim(domain->getNPop()),
    					 agentsForSim, &globalReward));
    	domain->ResetEpochEvals();
      }
      std::cout << "Test for " << globalReward.getName()
    		<< " Using policy: " << subobjectiveStrings[obj]
    		<< " Mean: " << mean(results)
    		<< " Stddev: " << stddev(results)
    		<< " Stderr: " << statstderr(results) << std::endl;
    }

    
    // Run tests for alignment
    
    std::vector<double> results;
    
    std::vector<Agent*> rands;
    domain = getDomain(expNode);
    for (size_t i = 0; i < domain->getNRovers(); i++) {
      rands.push_back(new RandomAgent());
    }
    domain->setTeam(rands);
    srand(time(NULL));
    for (size_t reps = 0; reps < 1000; reps++) {
      domain->InitialiseEpoch();

      results.push_back(domain->runSim(domain->createSim(domain->getNPop()),
    				       agentsForSim, &globalReward));
      domain->ResetEpochEvals();
      //std::cout << results[reps] << std::endl;
    }
    std::cout << "Test for " << globalReward.getName()
    	      << " Using random actions "
    	      << " Mean: " << mean(results)
    	      << " Stddev: " << stddev(results)
    	      << " Stderr: " << statstderr(results) << std::endl;
    
    results.clear();
    domain = getDomain(expNode);
    std::cout << "Using alignment to select action." << std::endl;

    results = alignmentGuidedAction(domain, &alignments, inds, teams,
    				    300, &globalReward, agentsForSim);
    std::cout << "Test for " << globalReward.getName()
    	      << " Using alignment to select action"
    	      << " Mean: " << mean(results)
    	      << " Stddev: " << stddev(results)
    	      << " Stderr: " << statstderr(results) << std::endl;

    
    //results.clear();
    domain = getDomain(expNode);
    std::cout << "Using alignment to select policy." << std::endl;

    results = alignmentGuidedPolicy(domain, &alignments, inds, teams,
    				    subobjectiveStrings.size(), 300,
    				    &globalReward, agentsForSim);
    std::cout << "Test for " << globalReward.getName()
    	      << " Using alignment to select policy"
    	      << " Mean: " << mean(results)
    	      << " Stddev: " << stddev(results)
    	      << " Stderr: " << statstderr(results) << std::endl;

    results.clear();
    domain = getDomain(expNode);
    std::cout << "Learning with alignment." << std::endl;

    results = alignmentLearning(domain, &alignments, inds, teams,
    				    100, &globalReward, agentsForSim);
    std::cout << "Test for " << globalReward.getName()
    	      << " Learning from alignment"
    	      << " Mean: " << mean(results)
    	      << " Stddev: " << stddev(results)
    	      << " Stderr: " << statstderr(results) << std::endl;

    // // // Directly using alignment

    // results.clear();
    // domain = getDomain(expNode);
    // std::cout << "Directly using alignment" << std::endl;
    // results = directAlignmentUse(domain, expNode, &globalReward,
    // 				 50, 0.4, objectives, 10);
    
    // std::cout << "Test for " << globalReward.getName()
    // 	      << " Using alignment directly "
    // 	      << " Mean: " << mean(results)
    // 	      << " Stddev: " << stddev(results)
    // 	      << " Stderr: " << statstderr(results) << std::endl;
    
    // Train and test G, D
    // results.clear();
    // domain = getDomain(expNode);
    // trainDomain(domain, expNode, expKey, fileDir, &g);
    // results = testDomain(domain, &globalReward, 50);
    // std::cout << "Test for " << globalReward.getName()
    // 	      << " Using policies trained with " << g.getName()
    // 	      << " Mean: " << mean(results)
    // 	      << " Stddev: " << stddev(results)
    // 	      << " Stderr: " << statstderr(results) << std::endl;

    // results.clear();
    // domain = getDomain(expNode);
    // trainDomain(domain, expNode, expKey, fileDir, &d);
    // results = testDomain(domain, &globalReward, 50);
    // std::cout << "Test for " << globalReward.getName()
    // 	      << " Using policies trained with " << d.getName()
    // 	      << " Mean: " << mean(results)
    // 	      << " Stddev: " << stddev(results)
    // 	      << " Stderr: " << statstderr(results) << std::endl;
    
    // // Train and test transfer learning
    // for (const auto& transferObjective : transferLearningObjectives) {
    //   results.clear();
    //   domain = getDomain(expNode);
    //   trainDomain(domain, expNode, transferObjective->getName(),
    // 		  fileDir, transferObjective);
    //   results = testDomain(domain, transferObjective, 100);
    //   std::cout << "Transfer Learning. Test for " << transferObjective->getName()
    // 		<< " Mean: " << mean(results)
    // 		<< " Stddev: " << stddev(results)
    // 		<< " Stderr: " << statstderr(results) << std::endl;
    // }

    std::cout << "Completed tests for objective" << expKey << std::endl;
  }

  return 0 ;
}
