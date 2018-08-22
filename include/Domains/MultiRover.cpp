#include "MultiRover.h"

MultiRover::MultiRover(vector<double> w, size_t numSteps, size_t numPop, size_t
		       numPOIs, Fitness f, size_t rovs, int c, AgentType t)
  : world(w), nSteps(numSteps), nPop(numPop), nPOIs(numPOIs), nRovers(rovs),
    coupling(c), fitness(f), outputEvals(false), outputTrajs(false),
    outputQury(false), outputBlf(false), gPOIObs(false), type(t), verbose(true),
    biasStart(true) {

  initRovers();
}

// MultiRover::MultiRover(vector<double> w, size_t numSteps, size_t
// 		       numPOIs, size_t rovs,
// 		       vector< vector<NeuralNet*> > nets, vector<vector<size_t>> inds,
// 		       Alignments* alignmentMap)
//   : world(w), nSteps(numSteps), nPop(1), nPOIs(numPOIs), nRovers(rovs),
//     coupling(1), fitness(Fitness::G), outputEvals(false), outputTrajs(false),
//     outputQury(false), outputBlf(false), gPOIObs(false), verbose(true),
//     biasStart(true) {

//   size_t nOut = inds.size();
//   for (size_t i = 0; i < nRovers; i++) {
//     vector<NeuralNet*> netsAgent;
//     for (size_t j = 0; j < nets.size(); j++) {
//       int netI = nets[j].size() > i ? i : 0;
//       netsAgent.push_back(nets[j][netI]);
//     }
//     roverTeam.push_back(new AlignmentAgent(netsAgent, alignmentMap, inds));
//     type = AgentType::C;
//   }
// }

MultiRover::~MultiRover() {
  // for (size_t i = 0; i < nRovers; i++){
  //   delete roverTeam[i] ;
  //   roverTeam[i] = 0 ;
  // }
  if (outputEvals)
    evalFile.close() ;
  if (outputTrajs){
    trajFile.close() ;
    POIFile.close() ;
  }
}

void MultiRover::initRovers() {
  roverTeam.clear();
  
  for (size_t i = 0; i < nRovers; i++) {
    if (type == AgentType::R) {
      roverTeam.push_back(new Rover(nSteps, nPop, fitness));
    } else if (type == AgentType::P) {
      roverTeam.push_back(new OnlyPOIRover(nSteps, nPop, fitness));
      setCoupling(1);
    } else if (type == AgentType::A) {
      roverTeam.push_back(new TeamFormingAgent(nSteps, nPop, fitness, coupling));
    } else if (type == AgentType::E) {
      roverTeam.push_back(new ExploringAgent(nSteps, nPop, fitness, coupling));
    }
  }
}
void MultiRover::InitialiseEpoch(){
  double rangeX = world[1] - world[0] ;
  double rangeY = world[3] - world[2] ;
  
  initialXYs.clear() ;
  initialPsis.clear() ;
  for (size_t i = 0; i < nRovers; i++){
    Vector2d initialXY ;
    if (type == AgentType::A || !biasStart) {
      initialXY(0) = rand_interval(world[0],world[1]);
      initialXY(1) = rand_interval(world[2],world[3]);
    } else {
      initialXY(0) = rand_interval(world[0]+rangeX/3.0,world[1]-rangeX/3.0);
      initialXY(1) = rand_interval(world[2]+rangeY/3.0,world[3]-rangeX/3.0);
    }
    double initialPsi = rand_interval(-PI,PI) ;
    initialXYs.push_back(initialXY) ;
    initialPsis.push_back(initialPsi) ;
  }
  
  // POI locations and values in global frame (restricted to within outer regions of 9 grid)
  POIs.clear() ;
  for (size_t p = 0; p < nPOIs; p++){
    Vector2d xy ;
    double x, y ;
    bool accept = false ;
    while (!accept){
      x = rand_interval(world[0],world[1]) ;
      y = rand_interval(world[2],world[3]) ;
      
      accept = !(x > world[0]+rangeX/3.0 &&
		 x < world[1]-rangeX/3.0 &&
		 y > world[2]+rangeY/3.0 &&
		 y < world[3]-rangeX/3.0);
    }
    
    xy(0) = x ; // x location
    xy(1) = y ; // y location
    double v = rand_interval(1,10) ; // value
    POIs.push_back(Target(xy, v, coupling));
  }
}

void MultiRover::InitialiseEpochFromVectors(vector<Target> targets,
					    vector<Vector2d> xys) {
  initialXYs = xys;
  POIs = targets;

  initialPsis.clear();
  for (size_t i = 0; i < nRovers; i++) {
    double initialPsi = rand_interval(-PI,PI);
    initialPsis.push_back(initialPsi);
  }
}

void MultiRover::InitialiseEpochFromOtherDomain(MultiRover* domain) {
  vector< State > otherStates = domain->getInitialStates();
  initialXYs.clear();
  initialPsis.clear();
  POIs.clear();

  for (const auto& state : otherStates) {
    initialXYs.push_back(state.pos());
    initialPsis.push_back(state.psi());
  }

  POIs = domain->getPOIs();
  for (auto& p : POIs) {
    p.ResetTarget();
  }
}


vector< vector<size_t> > MultiRover::RandomiseTeams(size_t n){
  vector< vector<size_t> > teams ;
  vector<size_t> order ;
  for (size_t i = 0; i < n; i++) {
    order.push_back(i) ;
  }
  
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() ;

  for (size_t j = 0; j < nRovers; j++){
    shuffle (order.begin(), order.end(), std::default_random_engine(seed)) ;
    teams.push_back(order) ;
  }
  
  return teams ;
}

vector<Env*> MultiRover::createEnvs(vector< vector< Agent* > > agents,
				    vector<string> labels, size_t teamSize) {
  vector<Env*> envs;
  vector< vector< Agent* > > dupAgents = duplicateAll(agents);
  
  for (size_t i = 0; i < agents.size(); i++) {
    envs.push_back(new Env(world, dupAgents[i], POIs, teamSize, labels[i]));
  }

  return envs;
}

vector< vector< Agent* > > MultiRover::duplicateAll(vector< vector< Agent* > > agents) {
  size_t maxsize = 0;
  for (const auto& i : agents) {
    if (i.size() > maxsize) maxsize = i.size();
  }

  vector< vector< Agent* > > allAgents;
  for (auto& i : agents) {
    allAgents.push_back(duplicate(i, maxsize));
  }

  return allAgents;
}

vector< Agent* > MultiRover::duplicate(vector< Agent* > agents, size_t n) {
  if (agents.size() == n) return agents;

  for (size_t i = 0; i < n - agents.size(); i=i) {
    agents.push_back(agents[i]->copyAgent());
  }

  return agents;
}

double MultiRover::runSim(Env* env, vector< size_t > teamIndex, Objective* o) {
  for (size_t t = 0; t < nSteps; t++) {
    vector< State > jointState = env->step(teamIndex);
    if (outputTrajs) {
      printJointState(jointState);
    }
  }

  return o->reward(env);
}

void MultiRover::init() {
  for (size_t a = 0; a < initialXYs.size(); a++) {
    State s(initialXYs[a], 0);
    roverTeam[a]->initialiseNewLearningEpoch(s, POIs);
  }
}
Env* MultiRover::createSim(size_t teamSize) {

  // std::cout << "MultiRover::createSim" << std::endl;
  // for (auto& i : roverTeam) {
  //   std::cout << i->getCurrentState() <<std::endl;
  // }
  Env* env = new Env(world, roverTeam, POIs, teamSize);

  //std::cout << "Create sim 2" << std::endl;
  vector< State > initState;

  for (size_t j = 0; j < nRovers; j++) {
    State s(initialXYs[j], initialPsis[j]);
    initState.push_back(s);
  }


  //std::cout << "Create sim 3" << std::endl;
  env->init(initState);

  //std::cout << "Create sim 4" << std::endl;
  return env;
}

double MultiRover::reward(Objective* o) {
  vector< State > currentState = getCurrentJointState();
  vector < vector<State> > states = {currentState};

  return reward(o, states);
}

double MultiRover::reward(Objective* o, vector< vector < State > > jointStates) {
  Env* env = new Env(world, roverTeam, POIs, getNPop());

  if (jointStates.size() < 1) {
    return 0;
  }

  env->init(jointStates[0]);

  for (size_t js = 1; js < jointStates.size(); js++) {
    env->applyStep(jointStates[js]);
  }

  return o->reward(env);
}

vector< State > MultiRover::getCurrentJointState() {
  vector< State > currentState;
  for (auto& rover : roverTeam) {
    currentState.push_back(rover->getCurrentState());
  }

  return currentState;
}
vector< NeuralNet* > MultiRover::getBestNNTeam(Objective* o) {
  vector< vector<size_t> > teams = RandomiseTeams(nPop);


  Env* env = createSim(nPop);

  double maxEval = 0, currentEval = 0;
  vector< size_t > netEachAgentUses;
  vector< NeuralNet* > bestNets;

  for (size_t i = 0; i < nPop; i++) {
    netEachAgentUses.clear();
    vector< State > jointState;

    for (size_t rov = 0; rov < nRovers; rov++) {
      netEachAgentUses.push_back(teams[rov][i]);
    }

    currentEval = runSim(env, netEachAgentUses, o);
    env->reset();

    if (currentEval >= maxEval) {
      bestNets.clear();
      maxEval = currentEval;

      for (size_t rov = 0; rov < nRovers; rov++) {
	Agent* rover = roverTeam[rov];
	size_t ind = netEachAgentUses[rov]; // teams[rov][i]
	bestNets.push_back(rover->GetNEPopulation()->GetNNIndex(ind));
      }
    }
  }

  return bestNets;
}

double MultiRover::SimulateEpoch(bool train, Objective* o) {
  //std::cout << "Simulate Epoch" << std::endl;
  size_t teamSize = train ? 2*nPop : nPop;
  // each row is the population for a single agent
  vector< vector<size_t> > teams = RandomiseTeams(teamSize) ; 
  if (outputTrajs) {
    printPOIs();
  }
  //std::cout << "Simulate Epoch" << std::endl;
  Env* env = createSim(teamSize);
  double maxEval = 0.0 ;
  vector< size_t > netEachAgentUses;
  double eval = 0.0;
  //std::cout << "Simulate Epoch" << std::endl;
  for (size_t i = 0; i < teamSize; i++) { // looping across the columns of 'teams'
    // Initialise world and reset rovers and POIs
    vector< State > jointState ;
    netEachAgentUses.clear();
    
    for (size_t j = 0; j < nRovers; j++) {
      netEachAgentUses.push_back(teams[j][i]);
    }

    if (outputTrajs && i == teamSize - 1) {
      printJointState(jointState);
      toggleAgentOutput(true);
    }

    eval = runSim(env, netEachAgentUses, o);
    env->reset();
    maxEval = max(eval, maxEval);
    // Assign fitness (G)

    vector<double> rewards = o->rewardV(env);
    //std::cout << "Simulate Epoch3" << std::endl;
    //std::cout << "rewards.size() " << rewards.size() << std::endl;
    //std::cout << "teams.size() " << teams.size() << std::endl;

    for (size_t j = 0; j < roverTeam.size(); j++) {
      //std::cout << "j " << j << std::endl;
      //std::cout << "rewards[j] " << rewards[j] << std::endl;
      //std::cout << "teams[j].size() " << teams[j].size() << std::endl;
      //std::cout << "teams[j][i] " << teams[j][i] << std::endl;
      roverTeam[j]->SetEpochPerformance(rewards[j], teams[j][i]);
    }
    //std::cout << "Simulate Epoch4" << std::endl;
    // If fitness is D -> eval is a vector
    // eval / maxEval average
    
    if (outputEvals) {
      evalFile << eval << "," ;
    }
  }

  delete env;
  if (outputEvals) {
    evalFile << std::endl;
  }

  if (verbose) {
    std::cout << "max achieved value: " << maxEval << "..." << std::endl;
  }
  //std::cout << "Simulate Epoch" << std::endl;
  if (train) {
    for (auto& rov : roverTeam) {
      rov->updateAgent();
    }
  }
  //std::cout << "Simulate Epoch end" << std::endl;
  return eval;
}

void MultiRover::EvolvePolicies(bool init){
  for (auto& rov : roverTeam) {
    rov->EvolvePolicies(init);
  }
}

void MultiRover::ResetEpochEvals(){
  for (auto& rov : roverTeam) {
    rov->ResetEpochEvals();
  }
}

// Wrapper for writing epoch evaluations to specified files
void MultiRover::OutputPerformance(std::string fileName){
  if (evalFile.is_open()) {
    evalFile.close();
  }
  
  evalFile.open(fileName.c_str(),std::ios::app);
  
  outputEvals = true;
}

// Wrapper for writing final trajectories to specified files
void MultiRover::OutputTrajectories(std::string tFile, std::string poiFile,
				    std::string tChoiceFile) {
  if (trajFile.is_open()) {
    trajFile.close();
  }
  trajFile.open(tFile.c_str(),std::ios::app) ;
  
  if (POIFile.is_open()) {
    POIFile.close();
  }
  POIFile.open(poiFile.c_str(),std::ios::app) ;

  for (auto& rov : roverTeam) {
    rov->openOutputFile(tChoiceFile);
  }
  
  outputTrajs = true ;
}

// Wrapper for writing final control policies to specified file
void MultiRover::OutputControlPolicies(std::string nnFile) {
  for (auto const& rover : roverTeam) {
    rover->OutputNNs(nnFile);
  }
}

// Wrapper for writing POMDP actions to specified file
void MultiRover::OutputQueries(char * A){
  // Filename to write to stored in A
  std::stringstream fileName ;
  fileName << A ;
  if (quryFile.is_open())
    quryFile.close() ;
  quryFile.open(fileName.str().c_str(),std::ios::app) ;
  
  outputQury = true ;
}

// Wrapper for writing POMDP beliefs to specified file
void MultiRover::OutputBeliefs(char * A){
	// Filename to write to stored in A
	std::stringstream fileName ;
  fileName << A ;
  if (blfFile.is_open())
    blfFile.close() ;
  blfFile.open(fileName.str().c_str(),std::ios::app) ;
  
  outputBlf = true ;
}

// Wrapper for writing running average of stepwiseD to specified file
void MultiRover::OutputAverageStepwise(char * A){
	// Filename to write to stored in A
	std::stringstream fileName ;
  fileName << A ;
  if (avgStepRFile.is_open())
    avgStepRFile.close() ;
  avgStepRFile.open(fileName.str().c_str(),std::ios::app) ;
  
  outputAvgStepR = true ;
}

void MultiRover::ExecutePolicies(string readFile, string storeTraj,
				 string storePOI, string storeEval,
				 size_t numIn, size_t numOut, size_t numHidden) {

  loadNNs(readFile, numIn, numHidden, numOut);
  

  if (getVerbose()) {
    std::cout << "Initialising test world..." << std::endl;
  }
  
  InitialiseEpoch() ;
  OutputPerformance(storeEval) ;
  OutputTrajectories(storeTraj, storePOI, "d") ;
  ResetEpochEvals() ;
  //SimulateEpoch(false) ; // simulate in test mode
}

void MultiRover::loadNNs(string nnFile, size_t numIn, size_t numHidden,
			 size_t numOut) {
  if (getVerbose()) {
    std::cout << "Reading in neural network policies from " << nnFile
	      << std::endl;
  }

  vector<NeuralNet*> loadedNN = NeuralNet::loadNNFromFile(nnFile, numIn, numHidden,
							  numOut);

  int k = 0;
  for (auto& rov : roverTeam) {
    NeuroEvo* rovNE = rov->GetNEPopulation();
    for (size_t i = 0; i < getNPop(); i++) {
      rovNE->GetNNIndex(i)->SetWeights(loadedNN[k]->GetWeightsA(),
				       loadedNN[k]->GetWeightsB());
      k++;
    }
  }

  for (size_t i = 0; i< loadedNN.size(); i++) {
    delete loadedNN[i];
    loadedNN[i] = 0;
  }
}

void MultiRover::loadNNsNeuralRover(vector<string> nnFiles, vector<size_t> nIns,
				    vector<size_t> nHiddens, vector<size_t> nOuts,
				    vector<vector<size_t>> inds) {

  vector< vector<NeuralNet*> > nets;
  for (size_t i = 0; i < nnFiles.size(); i++) {
    size_t nIn = nIns.size() == nnFiles.size() ? i : 0;
    size_t nH = nHiddens.size() == nnFiles.size() ? i : 0;
    size_t nOut = nOuts.size() == nnFiles.size() ? i : 0;
    vector<NeuralNet*> subNets = NeuralNet::loadNNFromFile(nnFiles[i], nIns[nIn],
							   nHiddens[nH], nOuts[nOut]);
    nets.push_back(subNets);
  }

  roverTeam.clear();
  
  size_t nOut = inds.size();
  for (size_t i = 0; i < getNRovers(); i++) {
    vector<NeuralNet*> netsAgent;
    for (auto& net : nets) {
      int netI = net.size() == getNRovers() ? i : 0;
      netsAgent.push_back(net[netI]);
    }
    roverTeam.push_back(new NeuralRover(getNSteps(), getNPop(), getFitness(),
					netsAgent, inds, nOut));
  }
}
void MultiRover::ExecutePolicies(string expFile, string novFile,
				 string storeTraj, string storePOI,
				 string storeEval, size_t numIn, size_t numOut,
				 size_t numHidden){

  if (verbose) {
    std::cout << "Reading out " << nPop << " expert NN control policies for"
	      << " each rover to test..." << std::endl;
  }
  
  vector<NeuralNet *> expLoadedNN = NeuralNet::loadNNFromFile(expFile,numIn,
							      numHidden,numOut);

  if (verbose) {
    std::cout << "Reading out " << nPop << " novice NN control policies "
	      << "for each rover to test..." << std::endl;
  }
  
  vector<NeuralNet *> novLoadedNN = NeuralNet::loadNNFromFile(novFile, numIn,
							      numHidden,numOut);
  
  // Assign control policies to rovers ;
  int k = 0 ;
  for (size_t i = 0; i < nRovers; i++){
    NeuroEvo * rovNE = roverTeam[i]->GetNEPopulation() ;
    if (k == 0){
      for (size_t j = 0; j < nPop; j++){
        rovNE->GetNNIndex(j)->SetWeights(novLoadedNN[k]->GetWeightsA(),novLoadedNN[k]->GetWeightsB()) ;
        k++ ;
      }
    }
    else {
      for (size_t j = 0; j < nPop; j++){
        rovNE->GetNNIndex(j)->SetWeights(expLoadedNN[k]->GetWeightsA(),expLoadedNN[k]->GetWeightsB()) ;
        k++ ;
      }
    }
  }
  
  // Initialise test world
  std::cout << "Initialising test world...\n" ;
  InitialiseEpoch() ;
  OutputPerformance(storeEval) ;
  OutputTrajectories(storeTraj, storePOI, "d") ;
  ResetEpochEvals() ;
  //SimulateEpoch(false) ; // simulate in test mode
  
  for (size_t i = 0; i < expLoadedNN.size(); i++){
    delete expLoadedNN[i] ;
    expLoadedNN[i] = 0 ;
  }
  for (size_t i = 0; i < novLoadedNN.size(); i++){
    delete novLoadedNN[i] ;
    novLoadedNN[i] = 0 ;
  }
}

vector<NeuralNet*> MultiRover::getNNTeam() {
  std::vector<NeuralNet*> nets;
  for (auto& rov : roverTeam) {
    nets.push_back(rov->GetNEPopulation()->GetNNIndex(0));
  }
  return nets;
}

void MultiRover::printPOIs() {
  for (auto const& poi : POIs) {
    POIFile << poi << std::endl;
  }
}
 
void MultiRover::printJointState(const vector<State> jointState) {
  for (const auto& s : jointState) {
    Vector2d vec = s.pos();
    trajFile << vec(0) << "," << vec(1) << ",";
  }
  trajFile << std::endl;
}

void MultiRover::toggleAgentOutput(bool toggle) {
  for (auto& rov : roverTeam) {
    rov->setOutputBool(toggle);
  }
}

void MultiRover::agentObserves(Vector2d xy, int time) {
  if (type == AgentType::A || type == AgentType::E) {
    for (const auto& other : roverTeam) {
      Vector2d otherLoc = other->getCurrentXY();
      if (!(otherLoc(0) == xy(0) && otherLoc(1) == xy(1))) {
	TeamFormingAgent* tA = (TeamFormingAgent*) other;
	tA->ObserveTarget(xy, time);
      }
    }
  } else {
    for (auto& poi : POIs) {
      poi.ObserveTarget(xy, time);
    }
 
    if (fitness == Fitness::D) {
      for (auto& tempPoi : tempPOIs) {
 	tempPoi.ObserveTarget(xy, time);
      }
    }
  }
}

void MultiRover::resetDomain() {
  resetAgents();
  resetPOIs();
}

void MultiRover::resetAgents() {
  if (type == AgentType::A || type == AgentType::E) {
    for (auto& rov : roverTeam) {
      TeamFormingAgent* tA = (TeamFormingAgent*) rov;
      tA->ResetTarget();
    }
  }
}

void MultiRover::resetPOIs() {
  for (auto& poi : POIs) {
    poi.ResetTarget();
  }
}


double MultiRover::calculateG() {
  double G = 0.0;

  for (auto& rover : roverTeam) {
    G += rover->getReward();
  }

  for (auto& poi : POIs) {
    G += calculatePOIValue(poi);
  }

  return G;
}

double MultiRover::calculatePOIValue(const Target& poi) {
  return poi.IsObserved() ? (poi.GetValue() / max(poi.GetNearestObs(), 1.0)) : 0.0;
}

double MultiRover::calculateStepwiseG() {
  double G = 0.0;
  
  for (const auto& poi : tempPOIs) {
    G += calculatePOIValue(poi);
  }
  
  return G;
}

std::ostream& operator<<(std::ostream &strm, const MultiRover &d) {
  double height = d.world[3] - d.world[2];
  double width  = d.world[1] - d.world[0];

  int h = floor(height);
  int w = floor(width);

  string output[h][w];
  
  double vals[h][w];
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      vals[i][j] = 0;
    }
  }

  for (const auto& poi : d.POIs) {
    Vector2d loc = poi.GetLocation();
    int fh = (int) (floor (loc(0)));
    int fw = (int) (floor (loc(1)));

    if (fh < 0 || fw < 0 || fw > width || fh > height) {
      continue;
    }
    // Look up 6.042. This is probably not needed. I forget why I put it in
    int ph = (fh + h) % h;
    int pw = (fw + w) % w;

    vals[ph][pw] += poi.GetValue();
    //strm << poi << std::endl;
  }
  
  for (const auto& rov : d.roverTeam) {
    //strm << *rov << std::endl;

    Vector2d loc = rov->getCurrentXY();
    int fh = (int) (floor (loc(0)));
    int fw = (int) (floor (loc(1)));
    if (fh < 0 || fw < 0 || fw > width || fh > height) {
      continue;
    }
    int ph = (fh + h) % h;
    int pw = (fw + w) % w;

    output[ph][pw] += "A";
  }

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      //strm << vals[i][j] << " ";
    }
    //strm << std::endl;
  }

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      if (vals[i][j] > 0) {
	output[i][j] += "P";
      }
      string out = output[i][j];
      out = (" " + out + " - ").substr(1,3);
      strm << out;
    }
    strm << std::endl;
  }
  
  return strm;
}

vector< State > MultiRover::getInitialStates() {
  vector< State > states;
  for (size_t i = 0; i < initialXYs.size(); i++) {
    State s(initialXYs[i], initialPsis[i]);
    states.push_back(s);
  }

  return states;
}
