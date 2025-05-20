#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include "game.h"
#include "cards.h"
#include "player.h"
#include "scoringsystem.h"

int main(void)
{
    int numPlayers;
    Player players[MAXPLAYERS];

    srand(time(NULL));

    numPlayers = setup(players);

    printf("\nPlaying with %d players: ", numPlayers);

    for (int i = 0; i < numPlayers; i++)
    {
        printf("%s", players[i].name);

        if(i < numPlayers - 2)
        {
            printf(", ");
        }
        else if(i == numPlayers - 2)
        {
            printf(" and ");
        }
    }
    printf("\n");

    playHand(numPlayers, players);

    for(int i = 0; i < numPlayers; i++)
    {
        if(players[i].hand)
        {
            freeHand(players[i].hand, 1);
        }
    }

    return 0;
}

void dealHand(Player players[], int numPlayers, Hand *deck, Hand *communityCards)
{
    // Reset player hands and make sure they're active
    for(int i = 0; i < numPlayers; i++)
    {
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

int playHand(int numPlayers, Player players[])
{
    int pot = 0;
    int cardsRevealed = 0;
    bool gameOver = false;
    int activePlayers = 0;

    // Count active players before starting
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status == ACTIVE || players[i].status == FOLDED) {
            if (players[i].credits > 0) {
                players[i].status = ACTIVE;
                activePlayers++;
            } else {
                players[i].status = NOT_PLAYING;
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

    dealHand(players, numPlayers, deck, communityCards);

    printf("\nStarting hand with %d active players\n", activePlayers);

    printf("\n--- Round 1 Predictions ---\n");
    gameOver = predictionRound(players, numPlayers, &pot, 1, communityCards, cardsRevealed);
    if (gameOver) {
        int result = endGame(players, numPlayers, pot, communityCards);
        freeHand(deck, 1);
        freeHand(communityCards, 1);
        return result;
    }

    cardsRevealed = 3;
    printf("\n--- The flop is: ");
    for (int i = 0; i < cardsRevealed; i++) {
        char cardStr[4];
        Card *c = getCard(communityCards, i);
        printCard(cardStr, c);
        printf("%s ", cardStr);
    }
    printf("---\n");

    printf("\n--- Round 2 Predictions ---\n");
    gameOver = predictionRound(players, numPlayers, &pot, 2, communityCards, cardsRevealed);
    if (gameOver) {
        int result = endGame(players, numPlayers, pot, communityCards);
        freeHand(deck, 1);
        freeHand(communityCards, 1);
        return result;
    }

    cardsRevealed = 4;
    printf("\n--- Turn revealed: ");
    char cardStr[4];
    Card *c = getCard(communityCards, 3);
    printCard(cardStr, c);
    printf("%s ---\n", cardStr);

    printf("\n--- Round 3 Predictions ---\n");
    gameOver = predictionRound(players, numPlayers, &pot, 3, communityCards, cardsRevealed);
    if (gameOver) {
        int result = endGame(players, numPlayers, pot, communityCards);
        freeHand(deck, 1);
        freeHand(communityCards, 1);
        return result;
    }

    cardsRevealed = 5;
    printf("\n--- River revealed: ");
    c = getCard(communityCards, 4);
    printCard(cardStr, c);
    printf("%s ---\n", cardStr);

    printf("\n--- Final Prediction Round ---\n");
    gameOver = predictionRound(players, numPlayers, &pot, 4, communityCards, cardsRevealed);

    int result = endGame(players, numPlayers, pot, communityCards);
    freeHand(deck, 1);
    freeHand(communityCards, 1);

    return result;
}

bool predictionRound(Player players[], int numPlayers, int *pot, int roundNum, Hand *communityCards, int cardsRevealed)
{
    int activePlayers = 0;
    char input;
    int prediction;
    char predictionAmount[50];

    // Count active players
    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status == ACTIVE) {
            activePlayers++;
        }
    }

    printf("Active players in round %d: %d\n", roundNum, activePlayers);

    if (activePlayers <= 1) {
        printf("Not enough active players to continue.\n");
        return true; // Activate game over sequence
    }

    for(int i = 0; i < numPlayers; i++)
    {
        if(players[i].status == ACTIVE)
        {
            char handStr[100];
            printHand(handStr, players[i].hand);
            printf("\n%s, these are your cards: %s", players[i].name, handStr);

            if(cardsRevealed > 0)
            {
                printf("\nCommunity cards: ");
                for(int j = 0; j < cardsRevealed; j++)
                {
                    char cardStr[4];
                    Card *c = getCard(communityCards, j);
                    printCard(cardStr, c);
                    printf("%s ", cardStr);
                }
                printf("\n");
            }

            printf("\nYou have %d credits, what would you like to do?\n", players[i].credits);
            printf("Call/Check - C\nRaise - R\nFold - F\n? : ");
            scanf(" %c", &input);
            clearInputBuffer();  // Clear the input buffer

            switch(input)
            {
                case 'C':
                case 'c':
                    // Hard coded amount for time being
                    prediction = 10;
                    if (prediction > players[i].credits)
                    {
                        prediction = players[i].credits;
                    }

                    printf("%s calls with %d credits\n", players[i].name, prediction);
                    players[i].credits -= prediction;
                    *pot += prediction;
                    break;

                case 'R':
                case 'r':
                    printf("How much would you like to raise? ");
                    scanf("%s", predictionAmount);
                    clearInputBuffer();  // Clear the input buffer
                    prediction = atoi(predictionAmount);

                    if (prediction <= 0 || prediction > players[i].credits) {
                        printf("Invalid bet amount. Defaulting to 20 credits.\n");
                        prediction = players[i].credits < 20 ? players[i].credits : 20;
                    }

                    printf("%s raises with %d credits\n", players[i].name, prediction);
                    players[i].credits -= prediction;
                    *pot += prediction;
                    break;

                case 'F':
                case 'f':
                    printf("%s folds\n", players[i].name);
                    players[i].status = FOLDED;
                    activePlayers--;

                    if (activePlayers <= 1) {
                        printf("Only one player left in the game.\n");
                        return true; // Game over - only one player left
                    }
                    break;

                default:
                    printf("Invalid choice, defaulting to call.\n");
                    prediction = 10;
                    if (prediction > players[i].credits) {
                        prediction = players[i].credits;
                    }
                    printf("%s calls with %d credits\n", players[i].name, prediction);
                    players[i].credits -= prediction;
                    *pot += prediction;
                    break;
            }

            printf("%s, you now have %d credits\n", players[i].name, players[i].credits);
        }
    }

    printf("\nRound %d complete. Pot contains %d credits.\n", roundNum, *pot);
    return false; // Game continues
}

int endGame(Player players[], int numPlayers, int pot, Hand* communityCards)
{
    int winner = -1;
    int activeCount = 0;
    char input;

    for (int i = 0; i < numPlayers; i++) {
        if (players[i].status == ACTIVE) {
            activeCount++;
            winner = i;
        }
    }

    if (activeCount == 0)
    {
        printf("\nNo active players left! The pot of %d credits goes to the house.\n", pot);
        return 0;
    }
    else if (activeCount == 1)
    {
        printf("\nGame over! %s wins the pot of %d credits by default as the only remaining player!\n",
               players[winner].name, pot);
        players[winner].credits += pot;
    }
    else {
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

        // Free the temporary community cards
        freeHand(communityCards, 1);
    }

    // Show final credits
    printf("\nFinal credits:\n");
    for (int i = 0; i < numPlayers; i++)
    {
        printf("%s: %d credits\n", players[i].name, players[i].credits);
    }

    // Mark players with no credits left as not playing
    for (int i = 0; i < numPlayers; i++)
    {
        if (players[i].credits <= 0)
        {
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
        playHand(numPlayers, players);
    }

    return 0;
}

// Convert player's hand and community cards into an array for processing
void combineCards(Player *player, Hand *communityCards, Card combined[], int *numCards) {
    *numCards = 0;

    // Add player's hole cards
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


