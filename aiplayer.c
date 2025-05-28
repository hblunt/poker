#include "neuralnetwork.h"
#include "player.h"
#include "game.h"
#include <string.h>

static NeuralNetwork *aiNetwork = NULL;

void initialiseAI()
{
    aiNetwork = loadNetwork("poker_ai.dat");

    if (!aiNetwork)
    {
        printf("Creating new neural network");
        {
            aiNetwork = createNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        }
    }
}

int aiMakeDecision(Player *player, Hand *communityCards, int pot, int currentBet, int numPlayers, int position)
{
    if (aiNetwork)
    {
        initialiseAI();
    }

    return makeDecision(aiNetwork, player, communityCards, pot, currentBet, numPlayers, position);
}

bool aiPredictionRound(Player players[],int numPlayers, int *pot, int roundNum, Hand* communityCards, int cardsRevealed, int startPosition, int *currentBetAmount)
{
    // Similar setup to normal round with some spins
    int activePlayers = 0;
    char input;
    int prediction;
    char predictionAmount[50];
    bool roundComplete = false;
    int currentPlayer = startPosition;
    int playersActed = 0;

    // Count active players
    for (int i = 0; i < numPlayers; i++)
    {
        if (players[i].status == ACTIVE)
        {
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

        // Check if this is an AI player (name starts with "AI_")
        if (strncmp(players[currentPlayer].name, "AI_", 3) == 0) {
            // AI decision
            int decision = aiMakeDecision(&players[currentPlayer], communityCards,
                                        *pot, *currentBetAmount, activePlayers, currentPlayer);

            printf("\n%s (AI) is thinking...\n", players[currentPlayer].name);

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
                    break;

                case 2: // Raise
                    // AI raises by 2x the big blind
                    int raiseAmount = *currentBetAmount + (2 * BIG_BLIND);
                    if (raiseAmount > players[currentPlayer].credits + players[currentPlayer].currentBet) {
                        // All-in
                        raiseAmount = players[currentPlayer].credits + players[currentPlayer].currentBet;
                    }

                    int amountToAdd = raiseAmount - players[currentPlayer].currentBet;
                    *currentBetAmount = raiseAmount;

                    printf("%s (AI) raises to %d credits\n", players[currentPlayer].name, raiseAmount);

                    players[currentPlayer].credits -= amountToAdd;
                    players[currentPlayer].currentBet = raiseAmount;
                    *pot += amountToAdd;

                    playersActed = 1; // Reset since we raised
                    break;
            }

            if (activePlayers <= 1) {
                return true;
            }

        }else {
            // Human player - existing code copy
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

            // Process human input (existing code)
            switch(input) {
                case 'C':
                case 'c':
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
                    } else {
                        prediction = 0;
                        printf("%s checks\n", players[currentPlayer].name);
                    }

                    players[currentPlayer].credits -= prediction;
                    players[currentPlayer].currentBet += prediction;
                    *pot += prediction;
                    break;

                case 'R':
                case 'r':
                    printf("How much would you like to raise to? (Current bet: %d) ", *currentBetAmount);
                    scanf("%s", predictionAmount);
                    clearInputBuffer();
                    prediction = atoi(predictionAmount);

                    if (prediction <= *currentBetAmount) {
                        printf("Raise must be greater than current bet. Defaulting to minimum raise.\n");
                        prediction = *currentBetAmount + 1;
                    }

                    int totalBet = prediction;
                    int amountToAdd = totalBet - players[currentPlayer].currentBet;

                    if (amountToAdd > players[currentPlayer].credits) {
                        amountToAdd = players[currentPlayer].credits;
                        totalBet = players[currentPlayer].currentBet + amountToAdd;
                        *currentBetAmount = totalBet;
                        printf("%s raises all-in to %d credits\n",
                               players[currentPlayer].name, totalBet);
                    } else {
                        *currentBetAmount = totalBet;
                        printf("%s raises to %d credits\n", players[currentPlayer].name, totalBet);
                    }

                    players[currentPlayer].credits -= amountToAdd;
                    players[currentPlayer].currentBet = totalBet;
                    *pot += amountToAdd;
                    playersActed = 1;
                    break;

                case 'F':
                case 'f':
                    printf("%s folds\n", players[currentPlayer].name);
                    players[currentPlayer].status = FOLDED;
                    activePlayers--;

                    if (activePlayers <= 1) {
                        return true;
                    }
                    break;

                default:
                    printf("Invalid choice. Please choose C (Call/Check), R (Raise), or F (Fold).\n");
                    continue;
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
    return false;
}

void saveAI()
{
    if (aiNetwork)
    {
        saveNetwork(aiNetwork, "poker_ai.dat");
        printf("AI network saved.\n");
    }
}

void cleanAI()
{
    if (aiNetwork)
    {
        freeNetwork(aiNetwork);
        aiNetwork = NULL;
    }
}
