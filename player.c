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
    }

    return numPlayers;
}
