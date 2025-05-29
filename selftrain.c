#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "neuralnetwork.h"
#include "aiplayer.h"
#include "player.h"
#include "game.h"
#include "cards.h"
#include "scoringsystem.h"
#include "selftrain.h"

// Create replay buffer
ReplayBuffer* createReplayBuffer(int capacity) {
    ReplayBuffer *rb = malloc(sizeof(ReplayBuffer));
    rb->buffer = malloc(capacity * sizeof(Experience));
    rb->capacity = capacity;
    rb->size = 0;
    rb->writeIndex = 0;
    return rb;
}

// Add experience to buffer
void addExperience(ReplayBuffer *rb, double *gameState, int action, double reward, int playerIndex, int handOutcome, int gameOutcome) {
    Experience *exp = &rb->buffer[rb->writeIndex];
    memcpy(exp->gameState, gameState, INPUT_SIZE * sizeof(double));
    exp->action = action;
    exp->reward = reward;
    exp->playerIndex = playerIndex;
    exp->handOutcome = handOutcome;
    exp->gameOutcome = gameOutcome;

    rb->writeIndex = (rb->writeIndex + 1) % rb->capacity;
    if (rb->size < rb->capacity) rb->size++;
}

// Determine winner based on hand strength
int determineWinner(Player players[], int numPlayers, Hand *communityCards) {
    int bestPlayer = -1;
    HandScore bestScore = {0};

    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status == ACTIVE) {
            Card combined[7];
            int numCards = 0;

            // Combine cards
            Card *current = players[i].hand->first;
            while (current && numCards < 2) {
                combined[numCards++] = *current;
                current = current->next;
            }

            current = communityCards->first;
            while (current && numCards < 7) {
                combined[numCards++] = *current;
                current = current->next;
            }

            HandScore score = findBestHand(combined, numCards);

            if (bestPlayer == -1 || compareHandScores(score, bestScore) > 0) {
                bestScore = score;
                bestPlayer = i;
            }
        }
    }

    return bestPlayer;
}


void generateTrainingData(double **inputs, double **outputs, int numSamples)
{
    srand(time(NULL));
    double gameState[INPUT_SIZE];

    for (int i = 0; i < numSamples; i++)
    {
        // Create more varied game states
        double handStrength = (double)rand() / RAND_MAX;
        gameState[0] = handStrength;

        double potOdds = (double)rand() / RAND_MAX;
        gameState[1] = potOdds;

        // Add more randomness and edge cases
        double stackSize = (double)rand() / RAND_MAX * 15 + 0.5;
        gameState[2] = stackSize;

        // Rest of the inputs...
        for (int j = 3; j < INPUT_SIZE; j++)
        {
            gameState[j] = (double)rand() / RAND_MAX;
        }

        for (int j = 0; j < INPUT_SIZE; j++)
        {
            inputs[i][j] = gameState[j];
        }

        // Improved strategy with more variety
        double fold = 0.0, call = 0.0, raise = 0.0;

        // Add some randomness to prevent overfitting
        double randomFactor = ((double)rand() / RAND_MAX - 0.5) * 0.2;

        if (handStrength < 0.2) {
            fold = 0.7 + randomFactor;
            call = 0.25;
            raise = 0.05;
        } else if (handStrength < 0.4) {
            fold = 0.4 + randomFactor;
            call = 0.5;
            raise = 0.1;
        } else if (handStrength < 0.7) {
            fold = 0.2;
            call = 0.6 + randomFactor;
            raise = 0.2;
        } else {
            fold = 0.05;
            call = 0.25;
            raise = 0.7 + randomFactor;
        }

        // Normalize and ensure valid probabilities
        double sum = fold + call + raise;
        if (sum <= 0) sum = 1.0;

        outputs[i][0] = fmax(0.01, fold / sum);
        outputs[i][1] = fmax(0.01, call / sum);
        outputs[i][2] = fmax(0.01, raise / sum);
    }
}

void trainBasicAI()
{
    printf("Training AI with a basic poker strategy...\n");

    // Create training data
    int numSamples = 1000;
    double **inputs = malloc(numSamples * sizeof(double*));
    double **outputs = malloc(numSamples * sizeof(double*));

    for (int i = 0; i < numSamples; i++)
    {
        inputs[i] = malloc(INPUT_SIZE * sizeof(double));
        outputs[i] = malloc(OUTPUT_SIZE * sizeof(double));
    }

    generateTrainingData(inputs, outputs, numSamples);

    NeuralNetwork *nn = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    train(nn, inputs, outputs, numSamples);

    saveNetwork(nn, "poker_ai.dat");
    printf("AI training complete. Data was saved.");

    // Cleanup
    for (int i = 0; i < numSamples; i++)
    {
        free(inputs[i]);
        free(outputs[i]);
    }
    free(inputs);
    free(outputs);
    freeNetwork(nn);
}

int selfPlayDecision(NeuralNetwork *nn, Player *player, Hand *communityCards, int pot, int currentBet, int numPlayers, int position, ReplayBuffer *rb, int playerIndex)
{
    double input[INPUT_SIZE];

    encodeGameState(player, communityCards, pot, currentBet, numPlayers, position, input);

    forwardpropagate(nn, input);

    // Random vs calculated decision
    static double epsilon = 0.3;  // Start higher
    epsilon = fmax(0.05, epsilon * 0.999);  // Decay gradually, minimum 5%
    int decision;

    if ((double)rand() / RAND_MAX < epsilon)
    {
        decision = rand() % OUTPUT_SIZE;
    } else {
        decision = 0;
        double bestProb = nn->outputLayer[0].value;
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            if (nn->outputLayer[i].value > bestProb) {
                bestProb = nn->outputLayer[i].value;
                decision = i;
            }
        }
    }

    addExperience(rb, input, decision, 0, playerIndex, 0, 0);

    return decision;
}

bool selfPlayPredictionRound(Player players[], int numPlayers, int *pot, int roundNum,Hand* communityCards, int cardsRevealed, int startPosition, int *currentBetAmount, NeuralNetwork **networks,
                             ReplayBuffer *rb, int *handDecisions)
{
    int activePlayers = 0;
    int currentPlayer = startPosition;
    int playersActed = 0;
    bool roundComplete = false;

    // Count active players
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status == ACTIVE) {
            activePlayers++;
        }
    }

    if (activePlayers <= 1) return true;

    while (!roundComplete) {
        if (players[currentPlayer].status != ACTIVE) {
            currentPlayer = (currentPlayer + 1) % numPlayers;
            continue;
        }

        if (playersActed >= activePlayers && allBetsMatched(players, numPlayers, *currentBetAmount)) {
            roundComplete = true;
            break;
        }

        int toCall = *currentBetAmount - players[currentPlayer].currentBet;

        // Each AI uses its own network
        int decision = selfPlayDecision(networks[currentPlayer], &players[currentPlayer],
                                       communityCards, *pot, *currentBetAmount,
                                       activePlayers, currentPlayer, rb, currentPlayer);

        // Track decision
        handDecisions[currentPlayer] = decision;

        // Execute decision
        switch(decision) {
            case 0: // Fold
                players[currentPlayer].status = FOLDED;
                activePlayers--;
                break;

            case 1: // Call/Check
                if (toCall > 0) {
                    int callAmount = (toCall > players[currentPlayer].credits) ?
                                   players[currentPlayer].credits : toCall;
                    players[currentPlayer].credits -= callAmount;
                    players[currentPlayer].currentBet += callAmount;
                    *pot += callAmount;
                }
                break;

            case 2: // Raise
                int raiseAmount = *currentBetAmount + (2 * BIG_BLIND);
                if (raiseAmount > players[currentPlayer].credits + players[currentPlayer].currentBet) {
                    raiseAmount = players[currentPlayer].credits + players[currentPlayer].currentBet;
                }

                int amountToAdd = raiseAmount - players[currentPlayer].currentBet;
                *currentBetAmount = raiseAmount;

                players[currentPlayer].credits -= amountToAdd;
                players[currentPlayer].currentBet = raiseAmount;
                *pot += amountToAdd;
                playersActed = 1;
                break;
        }

        if (activePlayers <= 1) return true;

        playersActed++;
        currentPlayer = (currentPlayer + 1) % numPlayers;
    }

    return false;
}

GameRecord playSelfPlayGame(NeuralNetwork **networks, int numPlayers, ReplayBuffer *rb) {
    GameRecord record = {0};
    record.numPlayers = numPlayers;

    Player players[MAXPLAYERS];

    // Initialize players
    for (int i = 0; i < numPlayers; i++) {
        sprintf(players[i].name, "AI %d", i);
        players[i].credits = STARTING_CREDITS;
        players[i].status = ACTIVE;
        players[i].hand = NULL;
        players[i].dealer = (i == 0);
        players[i].currentBet = 0;
        record.decisionCount[i] = 0;
    }

    int handsPlayed = 0;
    int maxHands = 100;
    int experienceStartIndex = rb->size;

    // Play until game ends
    while (handsPlayed < maxHands) {
        int activePlayers = 0;
        int lastActive = -1;

        for (int i = 0; i < numPlayers; i++) {
            if (players[i].credits > 0) {
                activePlayers++;
                lastActive = i;
            }
        }

        if (activePlayers <= 1) {
            record.winner = lastActive;
            break;
        }

        // Play one hand
        int handWinner = playSelfPlayHand(players, numPlayers, networks, rb, &record);
        handsPlayed++;
    }

    record.totalHands = handsPlayed;

    // Record final credits
    for (int i = 0; i < numPlayers; i++) {
        record.finalCredits[i] = players[i].credits;
    }

    // Find winner if not already determined
    if (record.winner == -1) {
        int maxCredits = 0;
        for (int i = 0; i < numPlayers; i++) {
            if (players[i].credits > maxCredits) {
                maxCredits = players[i].credits;
                record.winner = i;
            }
        }
    }

    // Update rewards in replay buffer based on game outcome
    updateRewards(rb, experienceStartIndex, &record);

    return record;
}

// Play single self-play hand
int playSelfPlayHand(Player players[], int numPlayers, NeuralNetwork **networks,
                     ReplayBuffer *rb, GameRecord *record) {
    int pot = 0;
    int cardsRevealed = 0;
    bool gameOver = false;
    int handDecisions[MAXPLAYERS] = {0};  // Track last decision per player
    int handStartCredits[MAXPLAYERS];

    // Save starting credits
    for (int i = 0; i < numPlayers; i++) {
        handStartCredits[i] = players[i].credits;
        if (players[i].credits > 0 && players[i].status != NOT_PLAYING) {
            players[i].status = ACTIVE;
        }
    }

    // Find dealer
    int currentDealer = -1;
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].dealer) {
            currentDealer = i;
            break;
        }
    }

    if (currentDealer == -1) {
        for (int i = 0; i < numPlayers; i++) {
            if (players[i].status == ACTIVE) {
                players[i].dealer = true;
                currentDealer = i;
                break;
            }
        }
    }

    // Create deck and deal
    Hand *deck = createDeck(1, 1);
    Hand *communityCards = createHand();

    // Blinds
    int smallBlindPos = findNextActivePlayer(players, numPlayers, currentDealer, 1);
    int bigBlindPos = findNextActivePlayer(players, numPlayers, smallBlindPos, 1);

    players[smallBlindPos].credits -= SMALL_BLIND;
    players[smallBlindPos].currentBet = SMALL_BLIND;
    pot += SMALL_BLIND;

    players[bigBlindPos].credits -= BIG_BLIND;
    players[bigBlindPos].currentBet = BIG_BLIND;
    pot += BIG_BLIND;

    dealHand(players, numPlayers, deck, communityCards);

    // Betting rounds
    int currentBetAmount = BIG_BLIND;
    int startPosition = findNextActivePlayer(players, numPlayers, bigBlindPos, 1);

    // Pre-flop
    gameOver = selfPlayPredictionRound(players, numPlayers, &pot, 1, communityCards,
                                      cardsRevealed, startPosition, &currentBetAmount,
                                      networks, rb, handDecisions);

    if (!gameOver) {
        // Flop
        resetCurrentBets(players, numPlayers);
        currentBetAmount = 0;
        cardsRevealed = 3;
        startPosition = findNextActivePlayer(players, numPlayers, currentDealer, 1);
        gameOver = selfPlayPredictionRound(players, numPlayers, &pot, 2, communityCards,
                                          cardsRevealed, startPosition, &currentBetAmount,
                                          networks, rb, handDecisions);
    }

    if (!gameOver) {
        // Turn
        resetCurrentBets(players, numPlayers);
        currentBetAmount = 0;
        cardsRevealed = 4;
        gameOver = selfPlayPredictionRound(players, numPlayers, &pot, 3, communityCards,
                                          cardsRevealed, startPosition, &currentBetAmount,
                                          networks, rb, handDecisions);
    }

    if (!gameOver) {
        // River
        resetCurrentBets(players, numPlayers);
        currentBetAmount = 0;
        cardsRevealed = 5;
        gameOver = selfPlayPredictionRound(players, numPlayers, &pot, 4, communityCards,
                                          cardsRevealed, startPosition, &currentBetAmount,
                                          networks, rb, handDecisions);
    }

    // Determine winner
    int handWinner = determineWinner(players, numPlayers, communityCards);
    if (handWinner >= 0) {
        players[handWinner].credits += pot;
    }

    // Calculate rewards for this hand
    for (int i = 0; i < numPlayers; i++) {
        if (handStartCredits[i] > 0) {  // Was active this hand
            double reward = (players[i].credits - handStartCredits[i]) / 100.0;  // Normalize

            // Bonus for winning the hand
            if (i == handWinner) {
                reward += 0.5;  // Significant bonus
            } else {
                reward -= 0.1;  // Small penalty for losing
            }

            // Update the last few experiences for this player with the reward
            int expCount = 0;
            for (int j = rb->size - 1; j >= 0 && expCount < 4; j--) {
                if (rb->buffer[j].playerIndex == i && rb->buffer[j].reward == 0) {
                    rb->buffer[j].reward = reward;
                    rb->buffer[j].handOutcome = (i == handWinner) ? 1 : 0;
                    expCount++;
                }
            }

            // Track decisions
            record->decisions[i][record->decisionCount[i]++] = handDecisions[i];
        }
    }

    // Move dealer
    players[currentDealer].dealer = false;
    int nextDealer = findNextActivePlayer(players, numPlayers, currentDealer, 1);
    if (nextDealer >= 0) {
        players[nextDealer].dealer = true;
    }

    // Cleanup
    freeHand(deck, 1);
    freeHand(communityCards, 1);
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].hand) {
            freeHand(players[i].hand, 1);
            players[i].hand = NULL;
        }
    }

    return handWinner;
}

// Update rewards based on game outcome
void updateRewards(ReplayBuffer *rb, int startIndex, GameRecord *record) {
    // Go through all experiences from this game
    for (int i = startIndex; i < rb->size; i++) {
        Experience *exp = &rb->buffer[i];

        // Add game outcome bonus
        if (exp->playerIndex == record->winner) {
            exp->reward += 1.0;  // Bonus for winning the game
            exp->gameOutcome = 1;
        } else {
            exp->reward -= 0.1;  // Small penalty for losing
            exp->gameOutcome = 0;
        }

        // Adjust rewards based on final position
        double positionBonus = record->finalCredits[exp->playerIndex] / 1000.0;
        exp->reward += positionBonus;
    }
}

// Train network from replay buffer
void trainFromExperience(NeuralNetwork *nn, ReplayBuffer *rb, int batchSize) {
    if (rb->size < batchSize) return;

    double totalError = 0;

    // Random sampling from replay buffer
    for (int i = 0; i < batchSize; i++) {
        int index = rand() % rb->size;
        Experience *exp = &rb->buffer[index];

        // Forward pass
        forwardpropagate(nn, exp->gameState);

        // Create target output
        double target[OUTPUT_SIZE];
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            target[j] = nn->outputLayer[j].value;
        }

        // Replace lines 444-447 with proper Q-learning update
        double learningRate = 0.1;  // Increased learning rate
        double discount = 0.9;
        target[exp->action] = exp->reward + discount * target[exp->action];

        // Ensure the target doesn't become extreme
        if (target[exp->action] > 1.0) target[exp->action] = 1.0;
        if (target[exp->action] < 0.0) target[exp->action] = 0.0;

        // Ensure probabilities sum to 1 and are valid
        double sum = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (target[j] < 0) target[j] = 0;
            sum += target[j];
        }
        if (sum > 0) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                target[j] /= sum;
            }
        }

        // Backpropagate
        backpropagate(nn, target);
        updateWeights(nn);

        // Track error
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            double diff = target[j] - nn->outputLayer[j].value;
            totalError += diff * diff;
        }
    }
}

// Main self-play training function
void selfPlayTraining(int numGames, int numPlayers) {
    printf("\n--- SELF-PLAY TRAINING ---\n");
    printf("Training %d AI players for %d games...\n", numPlayers, numGames);

    // Create neural networks for each player
    NeuralNetwork **networks = malloc(numPlayers * sizeof(NeuralNetwork*));
    for (int i = 0; i < numPlayers; i++) {
        networks[i] = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    }

    // Create shared replay buffer
    ReplayBuffer *rb = createReplayBuffer(100000);

    // Statistics
    int wins[MAXPLAYERS] = {0};
    double avgCredits[MAXPLAYERS] = {0};

    clock_t startTime = clock();

    // Training loop
    for (int game = 0; game < numGames; game++) {
        GameRecord record = playSelfPlayGame(networks, numPlayers, rb);

        // Update statistics
        wins[record.winner]++;
        for (int i = 0; i < numPlayers; i++) {
            avgCredits[i] = (avgCredits[i] * game + record.finalCredits[i]) / (game + 1);
        }

        // Train all networks from shared experience
        if (rb->size > 100 && game % 10 == 0) {  // Train every 10 games
            for (int i = 0; i < numPlayers; i++) {
                trainFromExperience(networks[i], rb, 32);  // Batch size 32
            }
        }

        // Progress update
        if (game % 100 == 0 && game > 0) {
            printf("\nProgress: %d/%d games\n", game, numGames);
            printf("Win distribution: ");
            for (int i = 0; i < numPlayers; i++) {
                printf("AI_%d: %.1f%% ", i, (wins[i] * 100.0) / game);
            }
            printf("\nAvg credits: ");
            for (int i = 0; i < numPlayers; i++) {
                printf("%.0f ", avgCredits[i]);
            }
            printf("\n");
        }
    }

    clock_t endTime = clock();
    double elapsed = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;

    // Final statistics
    printf("\n--- TRAINING COMPLETE ---\n");
    printf("Time elapsed: %.2f seconds (%.2f games/sec)\n", elapsed, numGames / elapsed);
    printf("\nFinal win rates:\n");
    for (int i = 0; i < numPlayers; i++) {
        printf("AI %d: %.1f%%\n", i, (wins[i] * 100.0) / numGames);
    }

    // Find best performing network
    int bestPlayer = 0;
    int maxWins = wins[0];
    for (int i = 1; i < numPlayers; i++) {
        if (wins[i] > maxWins) {
            maxWins = wins[i];
            bestPlayer = i;
        }
    }

    printf("\nBest performing AI: AI %d with %.1f%% win rate\n",
           bestPlayer, (wins[bestPlayer] * 100.0) / numGames);

    // Save the best network
    saveNetwork(networks[bestPlayer], "poker_ai_selfplay.dat");
    printf("Saved best network to poker_ai_selfplay.dat\n");

    // Optional: Save all networks
    char filename[50];
    for (int i = 0; i < numPlayers; i++) {
        sprintf(filename, "poker_ai_player_%d.dat", i);
        saveNetwork(networks[i], filename);
    }
    printf("Saved all individual networks\n");

    // Cleanup
    for (int i = 0; i < numPlayers; i++) {
        freeNetwork(networks[i]);
    }
    free(networks);
    free(rb->buffer);
    free(rb);
}

// Advanced training with different strategies
void advancedSelfPlayTraining() {
    int choice;

    printf("\n--- ADVANCED SELF-PLAY TRAINING ---\n");
    printf("1. Quick training (100 games, 4 players)\n");
    printf("2. Standard training (1000 games, 4 players)\n");
    printf("3. Intensive training (5000 games, 6 players)\n");
    printf("4. Custom training\n");
    printf("Choice: ");
    scanf("%d", &choice);
    clearInputBuffer();

    int numGames, numPlayers;

    switch(choice) {
        case 1:
            numGames = 100;
            numPlayers = 4;
            break;
        case 2:
            numGames = 1000;
            numPlayers = 4;
            break;
        case 3:
            numGames = 5000;
            numPlayers = 6;
            break;
        case 4:
            printf("Number of training games: ");
            scanf("%d", &numGames);
            printf("Number of AI players (2-8): ");
            scanf("%d", &numPlayers);
            clearInputBuffer();
            break;
        default:
            return;
    }

    selfPlayTraining(numGames, numPlayers);
}
