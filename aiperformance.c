#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "neuralnetwork.h"
#include "aiplayer.h"
#include "player.h"
#include "game.h"
#include "cards.h"

// Statistics structure
typedef struct {
    int gamesPlayed;
    int wins[MAXPLAYERS];
    int totalHands[MAXPLAYERS];
    int handsWon[MAXPLAYERS];
    int folds[MAXPLAYERS];
    int calls[MAXPLAYERS];
    int raises[MAXPLAYERS];
    double totalWinnings[MAXPLAYERS];
    double avgPotSize;
    int bankruptcies[MAXPLAYERS];
} GameStats;

int determineWinner(Player players[], int numPlayers, Hand *communityCards);

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


