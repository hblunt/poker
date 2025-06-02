
#include "neuralnetwork.h"
#include "player.h"
#include "game.h"
#include <string.h>

NeuralNetwork *aiNetwork = NULL;

// Initialize AI with priority loading system
void initialiseAI() {
    printf("Loading AI system...\n");
    
    // Priority 1: Evolutionary champion (ultimate AI)
    aiNetwork = loadNetwork("poker_ai_evolved_champion.dat");
    if (aiNetwork) {
        printf("LOADED: Evolutionary Champion AI\n");
        printf("This AI survived natural selection among 1000+ competitors!\n");
        return;
    }
    
    // Priority 2: Evolved champions (top 10)
    for (int i = 0; i < 10; i++) {
        char filename[100];
        sprintf(filename, "evolved_champion_%d.dat", i);
        aiNetwork = loadNetwork(filename);
        if (aiNetwork) {
            printf("LOADED: Evolved champion #%d from evolutionary training\n", i);
            return;
        }
    }
    
    // Priority 3: Regular evolved AI (from two-phase training)
    aiNetwork = loadNetwork("poker_ai_evolved.dat");
    if (aiNetwork) {
        printf("LOADED: Evolved AI (self-trained through reinforcement learning)\n");
        return;
    }
    
    // Priority 4: Evolved backups
    for (int i = 0; i < 6; i++) {
        char filename[100];
        sprintf(filename, "evolved_ai_%d.dat", i);
        aiNetwork = loadNetwork(filename);
        if (aiNetwork) {
            printf("LOADED: Evolved AI backup %d\n", i);
            return;
        }
    }
    
    // Priority 5: Bootstrap network
    aiNetwork = loadNetwork("poker_ai_bootstrap.dat");
    if (aiNetwork) {
        printf("LOADED: Bootstrap AI (basic rules only)\n");
        printf("Consider running training for better performance!\n");
        return;
    }
    
    // Priority 6: Legacy networks
    aiNetwork = loadNetwork("poker_ai_enhanced_monitored.dat");
    if (aiNetwork) {
        printf("LOADED: Legacy enhanced AI (old training method)\n");
        return;
    }
    
    aiNetwork = loadNetwork("poker_ai_selfplay_monitored.dat");
    if (aiNetwork) {
        printf("LOADED: Legacy self-play AI (old training method)\n");
        return;
    }
    
    // Last resort: Create fresh network
    printf("No trained AI networks found.\n");
    printf("Creating fresh neural network with random weights...\n");
    printf("WARNING: Untrained AI will make random decisions!\n");
    printf("TIP: Use option 2 or 3 to train the AI first.\n");
    aiNetwork = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
}

// AI decision making
int aiMakeDecision(Player *player, Hand *communityCards, int pot, int currentBet, int numPlayers, int position) {
    if (!aiNetwork) {
        initialiseAI();
    }
    
    int decision = makeEnhancedDecision(aiNetwork, player, communityCards, pot, currentBet, numPlayers, position);
    
    // Debug output
    printf("\nAI Debug (%s):\n", player->name);
    printf("  Fold: %.3f, Call: %.3f, Raise: %.3f\n",
           aiNetwork->outputLayer[0].value,
           aiNetwork->outputLayer[1].value,
           aiNetwork->outputLayer[2].value);
    printf("  Decision: %s\n", 
           decision == 0 ? "FOLD" : (decision == 1 ? "CALL/CHECK" : "RAISE"));
    
    return decision;
}

// Enhanced prediction round with opponent tracking
bool aiPredictionRound(Player players[], int numPlayers, int *pot, int roundNum, 
                      Hand* communityCards, int cardsRevealed, int startPosition, int *currentBetAmount) {
    resetRoundAggression();

    static bool initialized = false;
    if (!initialized) {
        initializeOpponentProfiles(numPlayers);
        initialized = true;
    }
    
    int activePlayers = 0;
    char input;
    int prediction;
    char predictionAmount[50];
    bool roundComplete = false;
    int currentPlayer = startPosition;
    int playersActed = 0;

    // Count active players
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status == ACTIVE) {
            activePlayers++;
        }
    }

    if (activePlayers <= 1) {
        return true;
    }

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

        // Check if this is an AI player
        if (strncmp(players[currentPlayer].name, "AI ", 3) == 0) {
            int decision = aiMakeDecision(&players[currentPlayer], communityCards,
                                        *pot, *currentBetAmount, activePlayers, currentPlayer);

            printf("\n%s (AI) is thinking...\n", players[currentPlayer].name);
            
            bool voluntaryAction = (decision != 0 || toCall == 0);
            updateOpponentProfile(currentPlayer, decision, voluntaryAction, 
                                players[currentPlayer].currentBet, *pot);

            switch(decision) {
                case 0: // Fold
                    printf("%s (AI) folds\n", players[currentPlayer].name);
                    players[currentPlayer].status = FOLDED;
                    activePlayers--;
                    break;

                case 1: // Call/Check
                    if (toCall > 0) {
                        prediction = toCall;
                        if (prediction > players[currentPlayer].credits) {
                            prediction = players[currentPlayer].credits;
                            printf("%s (AI) calls all-in with %d credits\n",
                                   players[currentPlayer].name, prediction);
                        } else {
                            printf("%s (AI) calls with %d credits\n",
                                   players[currentPlayer].name, prediction);
                        }
                    } else {
                        prediction = 0;
                        printf("%s (AI) checks\n", players[currentPlayer].name);
                    }

                    players[currentPlayer].credits -= prediction;
                    players[currentPlayer].currentBet += prediction;
                    *pot += prediction;
                    pause();
                    break;

                case 2: // Raise
                    int baseRaise = 2 * BIG_BLIND;
                    if (roundNum <= 1) baseRaise = 3 * BIG_BLIND;
                    
                    int raiseAmount = *currentBetAmount + baseRaise;
                    
                    if (raiseAmount > players[currentPlayer].credits + players[currentPlayer].currentBet) {
                        raiseAmount = players[currentPlayer].credits + players[currentPlayer].currentBet;
                    }

                    int amountToAdd = raiseAmount - players[currentPlayer].currentBet;
                    *currentBetAmount = raiseAmount;

                    printf("%s (AI) raises to %d credits\n", players[currentPlayer].name, raiseAmount);

                    players[currentPlayer].credits -= amountToAdd;
                    players[currentPlayer].currentBet = raiseAmount;
                    *pot += amountToAdd;

                    playersActed = 1;
                    pause();
                    break;
            }

            if (activePlayers <= 1) {
                return true;
            }

        } else {
            // Human player logic (unchanged from original)
            char handStr[100];
            printHand(handStr, players[currentPlayer].hand);
            printf("\n%s, these are your cards: %s", players[currentPlayer].name, handStr);

            if (cardsRevealed > 0) {
                printf("\nCommunity cards: ");
                for (int j = 0; j < cardsRevealed; j++) {
                    char cardStr[4];
                    Card *c = getCard(communityCards, j);
                    printCard(cardStr, c);
                    printf("%s ", cardStr);
                }
                printf("\n");
            }

            printf("\nYou have %d credits, current bet is %d",
                   players[currentPlayer].credits, *currentBetAmount);

            if (toCall > 0) {
                printf(", %d to call", toCall);
                printf("\nCall (%d) - C\nRaise - R\nFold - F\n? : ", toCall);
            } else {
                printf("\nCheck - C\nRaise - R\nFold - F\n? : ");
            }

            scanf(" %c", &input);
            clearInputBuffer();

            int humanDecision = -1;
            bool voluntaryAction = true;

            switch(input) {
                case 'C':
                case 'c':
                    humanDecision = 1;
                    if (toCall > 0) {
                        prediction = toCall;
                        if (prediction > players[currentPlayer].credits) {
                            prediction = players[currentPlayer].credits;
                            printf("%s calls all-in with %d credits\n",
                                   players[currentPlayer].name, prediction);
                        } else {
                            printf("%s calls with %d credits\n",
                                   players[currentPlayer].name, prediction);
                        }
                        voluntaryAction = true;
                    } else {
                        prediction = 0;
                        printf("%s checks\n", players[currentPlayer].name);
                        voluntaryAction = false;
                    }

                    players[currentPlayer].credits -= prediction;
                    players[currentPlayer].currentBet += prediction;
                    *pot += prediction;
                    pause();
                    break;

                case 'R':
                case 'r':
                    humanDecision = 2;
                    voluntaryAction = true;
                    printf("How much would you like to raise to? (Current bet: %d) ", *currentBetAmount);
                    scanf("%s", predictionAmount);
                    clearInputBuffer();
                    prediction = atoi(predictionAmount);

                    if (prediction <= *currentBetAmount) {
                        printf("Raise must be greater than current bet. Defaulting to minimum raise.\n");
                        prediction = *currentBetAmount + 1;
                    }

                    int totalBetAmount = prediction;
                    int amountToAdd = totalBetAmount - players[currentPlayer].currentBet;

                    if (amountToAdd > players[currentPlayer].credits) {
                        amountToAdd = players[currentPlayer].credits;
                        totalBetAmount = players[currentPlayer].currentBet + amountToAdd;
                        *currentBetAmount = totalBetAmount;
                        printf("%s raises all-in to %d credits\n",
                               players[currentPlayer].name, totalBetAmount);
                    } else {
                        *currentBetAmount = totalBetAmount;
                        printf("%s raises to %d credits\n", players[currentPlayer].name, totalBetAmount);
                    }

                    players[currentPlayer].credits -= amountToAdd;
                    players[currentPlayer].currentBet = totalBetAmount;
                    *pot += amountToAdd;
                    playersActed = 1;
                    pause();
                    break;

                case 'F':
                case 'f':
                    humanDecision = 0;
                    voluntaryAction = true;
                    printf("%s folds\n", players[currentPlayer].name);
                    players[currentPlayer].status = FOLDED;
                    activePlayers--;

                    if (activePlayers <= 1) {
                        return true;
                    }
                    pause();
                    break;

                default:
                    printf("Invalid choice. Please choose C (Call/Check), R (Raise), or F (Fold).\n");
                    continue;
            }
            
            if (humanDecision >= 0) {
                updateOpponentProfile(currentPlayer, humanDecision, voluntaryAction, 
                                    players[currentPlayer].currentBet, *pot);
            }
        }

        printf("%s, you now have %d credits\n", players[currentPlayer].name, players[currentPlayer].credits);

        if (players[currentPlayer].credits == 0 && players[currentPlayer].status == ACTIVE) {
            printf("%s is all-in!\n", players[currentPlayer].name);
        }

        playersActed++;
        currentPlayer = (currentPlayer + 1) % numPlayers;
    }

    printf("\nRound %d complete. Pot contains %d credits.\n", roundNum, *pot);
    pause();
    return false;
}

void saveAI() {
    if (aiNetwork) {
        saveNetwork(aiNetwork, "poker_ai_enhanced.dat");
        printf("Enhanced AI saved.\n");
    }
}

void cleanAI() {
    if (aiNetwork) {
        freeNetwork(aiNetwork);
        aiNetwork = NULL;
    }
}