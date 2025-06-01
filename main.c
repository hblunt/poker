#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include "game.h"
#include "cards.h"
#include "player.h"
#include "scoringsystem.h"
#include "neuralnetwork.h"
#include "aiplayer.h"
#include "selftrain.h"

int main(void) {
    int numPlayers;
    Player players[MAXPLAYERS];
    char choice;

    srand(time(NULL));

    printf("============================================\n");
    printf("=          GU HOLD'EM - ENHANCED AI        =\n");
    printf("============================================\n");
    printf("=                                          =\n");
    printf("=  1. Play against AI                      =\n");
    printf("=  2. Enhanced AI training (monitored)     =\n");
    printf("=  3. Self-play training (monitored)       =\n");
    printf("=  4. View training logs                   =\n");
    printf("=  5. Play without AI                      =\n");
    printf("=                                          =\n");
    printf("============================================\n");
    printf("Choice: ");
    scanf(" %c", &choice);
    clearInputBuffer();

    switch(choice) {
        case '1':
            printf("\n[AI] Loading AI system...\n");
            initialiseAI();
            numPlayers = setupWithAI(players);
            break;

        case '2':
            printf("\n[ENHANCED] Starting enhanced AI training with monitoring...\n");
            trainEnhancedBasicAIWithMonitoring();
            pause();
            initialiseAI();
            numPlayers = setupWithAI(players);
            break;

        case '3':
            printf("\n[SELF-PLAY] Starting self-play training with monitoring...\n");
            int games, aiPlayers;
            printf("Number of training games (recommended: 500-2000): ");
            scanf("%d", &games);
            printf("Number of AI players (2-6): ");
            scanf("%d", &aiPlayers);
            clearInputBuffer();
            
            if (games < 10) games = 100;
            if (aiPlayers < 2) aiPlayers = 2;
            if (aiPlayers > 6) aiPlayers = 6;
            
            enhancedSelfPlayTrainingWithMonitoring(games, aiPlayers);
            pause();
            initialiseAI();
            numPlayers = setupWithAI(players);
            break;

        case '4':
            printf("\n[LOGS] Viewing training logs...\n");
            viewTrainingLogs();
            pause();
            return main(); // Return to menu
            break;

        case '5':
            printf("\n[HUMAN] Setting up human-only game...\n");
            numPlayers = setup(players);
            break;

        default:
            printf("\n[ERROR] Invalid choice. Starting human-only game...\n");
            numPlayers = setup(players);
            break;
    }

    printf("\n[GAME] Playing with %d players: ", numPlayers);

    for (int i = 0; i < numPlayers; i++) {
        printf("%s", players[i].name);

        if(i < numPlayers - 2) {
            printf(", ");
        }
        else if(i == numPlayers - 2) {
            printf(" and ");
        }
    }
    printf("\n");

    // Check if we have AI players
    bool hasAI = false;
    for (int i = 0; i < numPlayers; i++) {
        if (strncmp(players[i].name, "AI ", 3) == 0) {
            hasAI = true;
            break;
        }
    }

    // Play the game
    if (hasAI) {
        printf("[AI] AI-enhanced game starting...\n");
        playHandAI(numPlayers, players);
    } else {
        printf("[HUMAN] Human-only game starting...\n");
        playHand(numPlayers, players);
    }

    // Cleanup
    for(int i = 0; i < numPlayers; i++) {
        if(players[i].hand) {
            freeHand(players[i].hand, 1);
        }
    }

    if (hasAI) {
        saveAI();
        cleanAI();
    }

    return 0;
}

// Training log viewing functions
void viewTrainingLogs() {
    printf("[LOGS] TRAINING LOGS VIEWER\n");
    printRepeatedChar('-', 50);
    printf("\n");
    
    // Check which log files exist
    FILE *file;
    
    printf("Available training logs:\n\n");
    
    // Check basic training log
    file = fopen("training_log.csv", "r");
    if (file) {
        printf("[OK] training_log.csv - Enhanced AI training results\n");
        fclose(file);
        
        // Show summary of basic training
        showBasicTrainingLog();
    } else {
        printf("[MISSING] training_log.csv - No enhanced training log found\n");
    }
    
    // Check self-play training log
    file = fopen("selfplay_progress.csv", "r");
    if (file) {
        printf("[OK] selfplay_progress.csv - Self-play training progress\n");
        fclose(file);
        
        // Show summary of self-play training
        showSelfPlayTrainingLog();
    } else {
        printf("[MISSING] selfplay_progress.csv - No self-play training log found\n");
    }
    
    printf("\n[TIP] You can open these .csv files in Excel or Google Sheets\n");
    printf("      to create graphs of the training progress!\n");
}

// Show summary of enhanced training log
void showBasicTrainingLog() {
    FILE *file = fopen("training_log.csv", "r");
    if (!file) return;
    
    printf("\n[SUMMARY] ENHANCED TRAINING SUMMARY:\n");
    printRepeatedChar('-', 35);
    printf("\n");
    
    char line[256];
    double firstLoss = 0, lastLoss = 0, bestLoss = 1000;
    double firstAcc = 0, lastAcc = 0, bestAcc = 0;
    int epochs = 0;
    double totalTime = 0;
    
    // Skip header
    fgets(line, sizeof(line), file);
    
    // Read training data
    while (fgets(line, sizeof(line), file)) {
        int epoch;
        double loss, accuracy, time, lr;
        if (sscanf(line, "%d,%lf,%lf,%lf,%lf", &epoch, &loss, &accuracy, &time, &lr) == 5) {
            if (epochs == 0) {
                firstLoss = loss;
                firstAcc = accuracy;
            }
            lastLoss = loss;
            lastAcc = accuracy;
            totalTime = time;
            
            if (loss < bestLoss) bestLoss = loss;
            if (accuracy > bestAcc) bestAcc = accuracy;
            
            epochs++;
        }
    }
    
    fclose(file);
    
    if (epochs > 0) {
        printf("Epochs completed: %d\n", epochs);
        printf("Total training time: %.1f seconds\n", totalTime);
        printf("Initial loss: %.6f -> Final loss: %.6f\n", firstLoss, lastLoss);
        printf("Best loss achieved: %.6f\n", bestLoss);
        printf("Initial accuracy: %.1f%% -> Final accuracy: %.1f%%\n", firstAcc * 100, lastAcc * 100);
        printf("Best accuracy: %.1f%%\n", bestAcc * 100);
        printf("Improvement: %.1f%% loss reduction\n", ((firstLoss - lastLoss) / firstLoss) * 100);
        
        if (lastLoss < firstLoss * 0.1) {
            printf("Status: [EXCELLENT] EXCELLENT training results!\n");
        } else if (lastLoss < firstLoss * 0.5) {
            printf("Status: [GOOD] GOOD training progress\n");
        } else {
            printf("Status: [WARNING] LIMITED improvement - may need more training\n");
        }
    }
}

// Show summary of self-play training log
void showSelfPlayTrainingLog() {
    FILE *file = fopen("selfplay_progress.csv", "r");
    if (!file) return;
    
    printf("\n[SUMMARY] SELF-PLAY TRAINING SUMMARY:\n");
    printRepeatedChar('-', 35);
    printf("\n");
    
    char line[512];
    int totalGames = 0;
    double finalWinRates[4] = {0};
    int finalBufferSize = 0;
    
    // Skip header
    fgets(line, sizeof(line), file);
    
    // Read to find the last entry
    while (fgets(line, sizeof(line), file)) {
        double wr0, wr1, wr2, wr3, avgCredits;
        int bufferSize;
        if (sscanf(line, "%d,%lf,%lf,%lf,%lf,%lf,%d", 
                   &totalGames, &wr0, &wr1, &wr2, &wr3, &avgCredits, &bufferSize) >= 6) {
            finalWinRates[0] = wr0;
            finalWinRates[1] = wr1;
            finalWinRates[2] = wr2;
            finalWinRates[3] = wr3;
            finalBufferSize = bufferSize;
        }
    }
    
    fclose(file);
    
    if (totalGames > 0) {
        printf("Games completed: %d\n", totalGames);
        printf("Experience samples: %d\n", finalBufferSize);
        printf("Final win rates:\n");
        
        int bestAI = 0;
        double bestWinRate = finalWinRates[0];
        
        for (int i = 0; i < 4; i++) {
            printf("  AI_%d: %.1f%%", i, finalWinRates[i]);
            if (finalWinRates[i] > bestWinRate) {
                bestWinRate = finalWinRates[i];
                bestAI = i;
            }
            if (i == bestAI) printf(" [BEST]");
            printf("\n");
        }
        
        // Calculate diversity
        double maxDiff = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                double diff = fabs(finalWinRates[i] - finalWinRates[j]);
                if (diff > maxDiff) maxDiff = diff;
            }
        }
        
        printf("Strategy diversity: %.1f%% (max win rate difference)\n", maxDiff);
        
        if (maxDiff < 10) {
            printf("Status: [BALANCED] BALANCED strategies - good training!\n");
        } else if (maxDiff < 25) {
            printf("Status: [MODERATE] MODERATE variation - decent training\n");
        } else {
            printf("Status: [WARNING] HIGH variation - one AI may be dominating\n");
        }
    }
}

void clearScreen()
{
    #ifdef _WIN32
             system("cls");
         #else
             system("clear");
         #endif
}

void pause() {
    printf("\nPress Enter to continue...");
    getchar();
    clearScreen();
}

int playHandAI(int numPlayers, Player players[])
{
    int pot = 0;
    int cardsRevealed = 0;
    bool gameOver = false;
    int activePlayers = 0;
    int currentDealer = players[0].dealer ? 0 : -1;

    // Count active players before starting and find current dealer
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status == ACTIVE || players[i].status == FOLDED) {
            if (players[i].credits > 0) {
                players[i].status = ACTIVE;
                activePlayers++;
            } else {
                players[i].status = NOT_PLAYING;
            }
        }

        if (players[i].dealer) {
            currentDealer = i;
        }
    }

    // If no dealer is set, assign to the first active player
    if (currentDealer == -1) {
        for (int i = 0; i < numPlayers; i++) {
            if (players[i].status == ACTIVE) {
                players[i].dealer = true;
                currentDealer = i;
                break;
            }
        }
    }

    if (activePlayers < 2) {
        printf("Not enough active players to start a game.\n");
        return 0;
    }

    // Create new deck and community cards
    Hand *deck = createDeck(1, 1);
    Hand *communityCards = createHand();

    // Calculate positions for small blind and big blind
    int smallBlindPos = findNextActivePlayer(players, numPlayers, currentDealer, 1);
    int bigBlindPos = findNextActivePlayer(players, numPlayers, smallBlindPos, 1);

    // Take blinds from players
    printf("\n%s is the dealer\n", players[currentDealer].name);
    printf("%s posts small blind of %d credits\n", players[smallBlindPos].name, SMALL_BLIND);
    int smallBlindAmount = (players[smallBlindPos].credits >= SMALL_BLIND) ? SMALL_BLIND : players[smallBlindPos].credits;
    players[smallBlindPos].credits -= smallBlindAmount;
    players[smallBlindPos].currentBet = smallBlindAmount;
    pot += smallBlindAmount;

    printf("%s posts big blind of %d credits\n", players[bigBlindPos].name, BIG_BLIND);
    int bigBlindAmount = (players[bigBlindPos].credits >= BIG_BLIND) ? BIG_BLIND : players[bigBlindPos].credits;
    players[bigBlindPos].credits -= bigBlindAmount;
    players[bigBlindPos].currentBet = bigBlindAmount;
    pot += bigBlindAmount;

    dealHand(players, numPlayers, deck, communityCards);

    printf("\nStarting hand with %d active players\n", activePlayers);
    pause();

    // Pre-flop
    printf("--- Pre-flop Predictions ---\n");
    int currentBetAmount = BIG_BLIND;
    int startPosition = findNextActivePlayer(players, numPlayers, bigBlindPos, 1);
    gameOver = aiPredictionRound(players, numPlayers, &pot, 1, communityCards, cardsRevealed, startPosition, &currentBetAmount);

    if (gameOver) {
        int result = endGame(players, numPlayers, pot, communityCards);
        freeHand(deck, 1);
        freeHand(communityCards, 1);
        return result;
    }

    // Same as playHand(), except with aiPredictionRound() instead
    resetCurrentBets(players, numPlayers);
    currentBetAmount = 0;

    clearScreen();
    cardsRevealed = 3;
    printf("\n--- The flop is: ");
    for (int i = 0; i < cardsRevealed; i++) {
        char cardStr[4];
        Card *c = getCard(communityCards, i);
        printCard(cardStr, c);
        printf("%s ", cardStr);
    }
    printf("---\n");
    pause();

    printf("\n--- Flop Predictions ---\n");
    startPosition = findNextActivePlayer(players, numPlayers, currentDealer, 1); // Start with player after dealer
    gameOver = aiPredictionRound(players, numPlayers, &pot, 2, communityCards, cardsRevealed, startPosition, &currentBetAmount);

    if (gameOver) {
        int result = endGame(players, numPlayers, pot, communityCards);
        freeHand(deck, 1);
        freeHand(communityCards, 1);
        return result;
    }

    // Reset player bets for new round
    resetCurrentBets(players, numPlayers);
    currentBetAmount = 0;

    clearScreen();
    cardsRevealed = 4;
    printf("\n--- Turn revealed: ");
    char cardStr[4];
    Card *c = getCard(communityCards, 3);
    printCard(cardStr, c);
    printf("%s ---\n", cardStr);
    pause();

    printf("\n--- Turn Predictions ---\n");
    startPosition = findNextActivePlayer(players, numPlayers, currentDealer, 1); // Start with player after dealer
    gameOver = aiPredictionRound(players, numPlayers, &pot, 3, communityCards, cardsRevealed, startPosition, &currentBetAmount);

    if (gameOver) {
        int result = endGame(players, numPlayers, pot, communityCards);
        freeHand(deck, 1);
        freeHand(communityCards, 1);
        return result;
    }

    // Reset player bets for new round
    resetCurrentBets(players, numPlayers);
    currentBetAmount = 0;

    clearScreen();
    cardsRevealed = 5;
    printf("\n--- River revealed: ");
    c = getCard(communityCards, 4);
    printCard(cardStr, c);
    printf("%s ---\n", cardStr);
    pause();

    printf("\n--- River Predictions ---\n");
    startPosition = findNextActivePlayer(players, numPlayers, currentDealer, 1); // Start with player after dealer
    gameOver =aiPredictionRound(players, numPlayers, &pot, 4, communityCards, cardsRevealed, startPosition, &currentBetAmount);

    int result = endGame(players, numPlayers, pot, communityCards);
    freeHand(deck, 1);
    freeHand(communityCards, 1);

    // Move dealer position to next active player for next hand
    players[currentDealer].dealer = false;
    int nextDealer = findNextActivePlayer(players, numPlayers, currentDealer, 1);
    players[nextDealer].dealer = true;

    return result;
}

void dealHand(Player players[], int numPlayers, Hand *deck, Hand *communityCards) {
    // Reset player hands and make sure they're active
    for(int i = 0; i < numPlayers; i++) {
        // Reset folded players back to active at start of new hand
        if (players[i].status == FOLDED && players[i].credits > 0) {
            players[i].status = ACTIVE;
        }

        // Only deal to players who are still in the game
        if (players[i].status != NOT_PLAYING) {
            if (players[i].hand) {
                freeHand(players[i].hand, 1);
            }
            players[i].hand = createHand();
        }
    }

    // Deal two cards to each active player
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < numPlayers; j++) {
            if (players[j].status == ACTIVE) {
                Card *c = getTop(deck);
                if(c) {
                    addCard(players[j].hand, c);
                }
            }
        }
    }

    // Deal five community cards
    for (int i = 0; i < 5; i++) {
        Card *c = getTop(deck);
        if (c) {
            addCard(communityCards, c);
        }
    }
}

int playHand(int numPlayers, Player players[]) {
    int pot = 0;
    int cardsRevealed = 0;
    bool gameOver = false;
    int activePlayers = 0;
    int currentDealer = players[0].dealer ? 0 : -1;

    // Count active players before starting and find current dealer
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status == ACTIVE || players[i].status == FOLDED) {
            if (players[i].credits > 0) {
                players[i].status = ACTIVE;
                activePlayers++;
            } else {
                players[i].status = NOT_PLAYING;
            }
        }

        if (players[i].dealer) {
            currentDealer = i;
        }
    }

    // If no dealer is set, assign to the first active player
    if (currentDealer == -1) {
        for (int i = 0; i < numPlayers; i++) {
            if (players[i].status == ACTIVE) {
                players[i].dealer = true;
                currentDealer = i;
                break;
            }
        }
    }

    if (activePlayers < 2) {
        printf("Not enough active players to start a game.\n");
        return 0;
    }

    // Create new deck and community cards
    Hand *deck = createDeck(1, 1);
    Hand *communityCards = createHand();

    // Calculate positions for small blind and big blind
    int smallBlindPos = findNextActivePlayer(players, numPlayers, currentDealer, 1);
    int bigBlindPos = findNextActivePlayer(players, numPlayers, smallBlindPos, 1);

    // Take blinds from players
    printf("\n%s is the dealer\n", players[currentDealer].name);
    printf("%s posts small blind of %d credits\n", players[smallBlindPos].name, SMALL_BLIND);
    int smallBlindAmount = (players[smallBlindPos].credits >= SMALL_BLIND) ? SMALL_BLIND : players[smallBlindPos].credits;
    players[smallBlindPos].credits -= smallBlindAmount;
    players[smallBlindPos].currentBet = smallBlindAmount;
    pot += smallBlindAmount;

    printf("%s posts big blind of %d credits\n", players[bigBlindPos].name, BIG_BLIND);
    int bigBlindAmount = (players[bigBlindPos].credits >= BIG_BLIND) ? BIG_BLIND : players[bigBlindPos].credits;
    players[bigBlindPos].credits -= bigBlindAmount;
    players[bigBlindPos].currentBet = bigBlindAmount;
    pot += bigBlindAmount;

    dealHand(players, numPlayers, deck, communityCards);

    printf("\nStarting hand with %d active players\n", activePlayers);
    pause();

    printf("\n--- Pre-flop Predictions ---\n");
    int currentBetAmount = BIG_BLIND;
    int startPosition = findNextActivePlayer(players, numPlayers, bigBlindPos, 1); // Start with player after big blind
    gameOver = predictionRound(players, numPlayers, &pot, 1, communityCards, cardsRevealed, startPosition, &currentBetAmount);

    if (gameOver) {
        int result = endGame(players, numPlayers, pot, communityCards);
        freeHand(deck, 1);
        freeHand(communityCards, 1);
        return result;
    }

    // Reset player bets for new round
    resetCurrentBets(players, numPlayers);
    currentBetAmount = 0;

    cardsRevealed = 3;
    printf("\n--- The flop is: ");
    for (int i = 0; i < cardsRevealed; i++) {
        char cardStr[4];
        Card *c = getCard(communityCards, i);
        printCard(cardStr, c);
        printf("%s ", cardStr);
    }
    printf("---\n");
    pause();

    printf("\n--- Flop Predictions ---\n");
    startPosition = findNextActivePlayer(players, numPlayers, currentDealer, 1); // Start with player after dealer
    gameOver = predictionRound(players, numPlayers, &pot, 2, communityCards, cardsRevealed, startPosition, &currentBetAmount);

    if (gameOver) {
        int result = endGame(players, numPlayers, pot, communityCards);
        freeHand(deck, 1);
        freeHand(communityCards, 1);
        return result;
    }

    // Reset player bets for new round
    resetCurrentBets(players, numPlayers);
    currentBetAmount = 0;

    clearScreen();
    cardsRevealed = 4;
    printf("\n--- Turn revealed: ");
    char cardStr[4];
    Card *c = getCard(communityCards, 3);
    printCard(cardStr, c);
    printf("%s ---\n", cardStr);
    pause();

    printf("\n--- Turn Predictions ---\n");
    startPosition = findNextActivePlayer(players, numPlayers, currentDealer, 1); // Start with player after dealer
    gameOver = predictionRound(players, numPlayers, &pot, 3, communityCards, cardsRevealed, startPosition, &currentBetAmount);

    if (gameOver) {
        int result = endGame(players, numPlayers, pot, communityCards);
        freeHand(deck, 1);
        freeHand(communityCards, 1);
        return result;
    }

    // Reset player bets for new round
    resetCurrentBets(players, numPlayers);
    currentBetAmount = 0;

    clearScreen();
    cardsRevealed = 5;
    printf("\n--- River revealed: ");
    c = getCard(communityCards, 4);
    printCard(cardStr, c);
    printf("%s ---\n", cardStr);
    pause();

    printf("\n--- River Predictions ---\n");
    startPosition = findNextActivePlayer(players, numPlayers, currentDealer, 1); // Start with player after dealer
    gameOver = predictionRound(players, numPlayers, &pot, 4, communityCards, cardsRevealed, startPosition, &currentBetAmount);

    int result = endGame(players, numPlayers, pot, communityCards);
    freeHand(deck, 1);
    freeHand(communityCards, 1);

    // Move dealer position to next active player for next hand
    players[currentDealer].dealer = false;
    int nextDealer = findNextActivePlayer(players, numPlayers, currentDealer, 1);
    players[nextDealer].dealer = true;

    return result;
}

// Find the next active player in the circle
int findNextActivePlayer(Player players[], int numPlayers, int currentPos, int offset) {
    int count = 0;
    int pos = currentPos;

    while (count < numPlayers) {
        pos = (pos + offset) % numPlayers;
        if (players[pos].status == ACTIVE) {
            return pos;
        }
        count++;
    }

    // If we get here, there's no active player (should never happen)
    return currentPos;
}

// Reset current bets for all players
void resetCurrentBets(Player players[], int numPlayers) {
    for (int i = 0; i < numPlayers; i++) {
        players[i].currentBet = 0;
    }
}

bool predictionRound(Player players[], int numPlayers, int *pot, int roundNum, Hand* communityCards, int cardsRevealed, int startPosition, int *currentBetAmount) {
    int activePlayers = 0;
    char input;
    int prediction;
    char predictionAmount[50];
    bool roundComplete = false;
    int currentPlayer = startPosition;
    int playersActed = 0;
    int playersAllIn = 0;

    // Count active players
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status == ACTIVE) {
            activePlayers++;
        }
    }

    printf("\nActive players in round %d: %d\n", roundNum, activePlayers);

    if (activePlayers <= 1) {
        printf("Not enough active players to continue.\n");
        return true; // Activate game over sequence
    }

    // Continue the betting round until all active players have acted and all bets are matched
    while (!roundComplete) {
        // Skip players who have folded or are not playing
        if (players[currentPlayer].status != ACTIVE) {
            currentPlayer = (currentPlayer + 1) % numPlayers;
            continue;
        }

        // Check if betting round is complete
        if (playersActed >= activePlayers && allBetsMatched(players, numPlayers, *currentBetAmount)) {
            roundComplete = true;
            break;
        }

        pause();
        printf("--- %s's Turn ---\n", players[currentPlayer].name);
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

        int toCall = *currentBetAmount - players[currentPlayer].currentBet;

        printf("\nYou have %d credits, current bet is %d", players[currentPlayer].credits, *currentBetAmount);

        if (toCall > 0) {
            printf(", %d to call", toCall);
            printf("\nCall (%d) - C\nRaise - R\nFold - F\n? : ", toCall);
        } else {
            printf("\nCheck - C\nRaise - R\nFold - F\n? : ");
        }

        scanf(" %c", &input);
        clearInputBuffer();  // Clear the input buffer

        switch(input) {
            case 'C':
            case 'c':
                clearScreen();
                if (toCall > 0) {
                    // Call
                    prediction = toCall;
                    if (prediction > players[currentPlayer].credits) {
                        // All-in
                        prediction = players[currentPlayer].credits;
                        printf("%s calls all-in with %d credits\n", players[currentPlayer].name, prediction);
                        playersAllIn++;
                    } else {
                        printf("%s calls with %d credits\n", players[currentPlayer].name, prediction);
                    }
                } else {
                    // Check
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
                clearInputBuffer();  // Clear the input buffer
                prediction = atoi(predictionAmount);

                // Validate raise amount
                if (prediction <= *currentBetAmount) {
                    printf("Raise must be greater than current bet. Defaulting to minimum raise (%d).\n", *currentBetAmount + 1);
                    prediction = *currentBetAmount + 1;
                }

                int totalBet = prediction;  // The total amount player is betting this round
                int amountToAdd = totalBet - players[currentPlayer].currentBet;  // How much more to add

                // Check if player has enough credits
                if (amountToAdd > players[currentPlayer].credits) {
                    // All-in
                    amountToAdd = players[currentPlayer].credits;
                    totalBet = players[currentPlayer].currentBet + amountToAdd;
                    *currentBetAmount = totalBet;  // Update current bet
                    clearScreen();
                    printf("%s raises all-in to %d credits\n", players[currentPlayer].name, totalBet);
                    playersAllIn++;
                } else {
                    *currentBetAmount = totalBet;  // Update current bet
                    clearScreen();
                    printf("%s raises to %d credits\n", players[currentPlayer].name, totalBet);
                }

                players[currentPlayer].credits -= amountToAdd;
                players[currentPlayer].currentBet = totalBet;
                *pot += amountToAdd;

                playersActed = 1;
                break;

            case 'F':
            case 'f':
                clearScreen();
                printf("%s folds\n", players[currentPlayer].name);
                players[currentPlayer].status = FOLDED;
                activePlayers--;

                if (activePlayers <= 1) {
                    printf("Only one player left in the game.\n");
                    return true; // Game over - only one player left
                }
                break;

            default:
                printf("Invalid choice. Please choose C (Call/Check), R (Raise), or F (Fold).\n");
                // Don't increment playersActed or move to next player
                continue;
        }

        printf("%s, you now have %d credits\n\n", players[currentPlayer].name, players[currentPlayer].credits);



        // Check if player is all-in
        if (players[currentPlayer].credits == 0 && players[currentPlayer].status == ACTIVE) {
            printf("%s is all-in!\n", players[currentPlayer].name);
            playersAllIn++;
        }

        playersActed++;
        currentPlayer = (currentPlayer + 1) % numPlayers;
    }

    printf("\nRound %d complete. Pot contains %d credits.\n", roundNum, *pot);
    return false; // Game continues
}

// Check if all active players have matched the current bet
bool allBetsMatched(Player players[], int numPlayers, int currentBet) {
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status == ACTIVE && players[i].currentBet < currentBet && players[i].credits > 0) {
            return false;
        }
    }
    return true;
}

int endGame(Player players[], int numPlayers, int pot, Hand* communityCards) {
    int winner = -1;
    int activeCount = 0;
    char input;

    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status == ACTIVE) {
            activeCount++;
            winner = i;
        }
    }

    if (activeCount == 0) {
        printf("\nNo active players left! The pot of %d credits goes to the house.\n", pot);
        return 0;
    } else if (activeCount == 1) {
        printf("\nGame over! %s wins the pot of %d credits by default as the only remaining player!\n",
               players[winner].name, pot);
        players[winner].credits += pot;
    } else {
        // Multiple active players - determine winner based on hand strength
        HandScore bestScores[MAXPLAYERS];
        char handDescriptions[MAXPLAYERS][100];
        Card combinedCards[7];
        int numCards;

        // Display community cards
        printf("Community cards: ");
        char handStr[100];
        printHand(handStr, communityCards);
        printf("%s\n", handStr);

        // Calculate best hand for each active player
        int highestScoreIndex = -1;
        for (int i = 0; i < numPlayers; i++) {
            if (players[i].status == ACTIVE) {
                // Combine player's cards with community cards
                combineCards(&players[i], communityCards, combinedCards, &numCards);

                // Find best 5-card hand
                bestScores[i] = findBestHand(combinedCards, numCards);

                // Get description of the hand
                describeHandScore(bestScores[i], handDescriptions[i]);

                // Display player's hand and best hand
                printf("\n%s's cards: ", players[i].name);
                printHand(handStr, players[i].hand);
                printf("%s\n", handStr);
                printf("Best hand: %s\n", handDescriptions[i]);

                // Keep track of highest score
                if (highestScoreIndex == -1 ||
                    compareHandScores(bestScores[i], bestScores[highestScoreIndex]) > 0) {
                    highestScoreIndex = i;
                }
            }
        }

        // Set the winner to the player with the highest score
        winner = highestScoreIndex;

        printf("\nGame over! %s wins with %s!\n",
               players[winner].name, handDescriptions[winner]);
        printf("%s wins the pot of %d credits!\n", players[winner].name, pot);
        players[winner].credits += pot;
    }

    // Show final credits
    printf("\nFinal credits:\n");
    for (int i = 0; i < numPlayers; i++) {
        printf("%s: %d credits\n", players[i].name, players[i].credits);
    }

    // Mark players with no credits left as not playing
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].credits <= 0) {
            printf("%s is out of credits and has been removed from the game.\n", players[i].name);
            players[i].status = NOT_PLAYING;
        }
    }

    // Ask all players if they want to continue
    printf("\nWould you like to play another hand? (Y/N): ");
    scanf(" %c", &input);
    clearInputBuffer();

    if (input == 'Y' || input == 'y') {
        // Reset player statuses for next hand
        for (int i = 0; i < numPlayers; i++) {
            if (players[i].status == FOLDED && players[i].credits > 0) {
                players[i].status = ACTIVE;
            }
        }

        // Play another hand

        bool hasAI = false;
        for (int i = 0; i < numPlayers; i++) {
            if (strncmp(players[i].name, "AI ", 3) == 0) {
                hasAI = true;
                break;
            }
        }

        if(hasAI)
        {
            playHandAI(numPlayers, players);
        } else {
            playHand(numPlayers, players);
        }
    }

    return 0;
}

// Convert player's hand and community cards into an array for processing
void combineCards(Player *player, Hand *communityCards, Card combined[], int *numCards) {
    *numCards = 0;

    // Add player's cards
    Card *current = player->hand->first;
    while (current != NULL) {
        combined[(*numCards)++] = *current;
        current = current->next;
    }

    // Add community cards
    current = communityCards->first;
    while (current != NULL) {
        combined[(*numCards)++] = *current;
        current = current->next;
    }
}