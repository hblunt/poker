
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "player.h"

// Helper function to clear input buffer
void clearInputBuffer() {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}

// Set up players for the game
int setup(Player players[]) {
    int numPlayers = 0;
    char input[500];

    while(numPlayers < 2 || numPlayers > MAXPLAYERS) {
        printf("Enter number of players (max 8): ");
        fgets(input, 500, stdin);
        sscanf(input, "%d", &numPlayers);
    }

    for (int i = 0; i < numPlayers; i++) {
        printf("Enter Player %d name: ", i+1);
        fgets(input, 500, stdin);
        input[strcspn(input, "\n")] = 0;
        strcpy(players[i].name, input);
        players[i].credits = STARTING_CREDITS;
        players[i].status = ACTIVE;
        players[i].hand = createHand();
        players[i].dealer = (i == 0) ? true : false; // First player is dealer initially
        players[i].currentBet = 0;
    }

    return numPlayers;
}

int setupWithAI(Player players[])
{
    int numPlayers = 0;
    int numAI = 0;
    char input[500];

    while(numPlayers < 2 || numPlayers > MAXPLAYERS) {
        printf("Enter number of players (max 8): ");
        fgets(input, 500, stdin);
        sscanf(input, "%d", &numPlayers);
    }

    printf("How many AI players? (0-%d): ", numPlayers - 1);
    fgets(input, 500, stdin);
    sscanf(input, "%d", &numAI);

    if (numAI >= numPlayers) {
        numAI = numPlayers - 1;
        printf("Setting %d AI players (need at least 1 human)\n", numAI);
    }

    // Set up human players first
    for (int i = 0; i < numPlayers - numAI; i++) {
        printf("Enter Player %d name: ", i+1);
        fgets(input, 500, stdin);
        input[strcspn(input, "\n")] = 0;
        strcpy(players[i].name, input);
        players[i].credits = STARTING_CREDITS;
        players[i].status = ACTIVE;
        players[i].hand = createHand();
        players[i].dealer = (i == 0) ? true : false;
        players[i].currentBet = 0;
    }

    // Set up AI players
    for (int i = numPlayers - numAI; i < numPlayers; i++) {
        sprintf(players[i].name, "AI Bot %d", i - (numPlayers - numAI) + 1);
        players[i].credits = STARTING_CREDITS;
        players[i].status = ACTIVE;
        players[i].hand = createHand();
        players[i].dealer = false;
        players[i].currentBet = 0;
    }

    return numPlayers;
}
