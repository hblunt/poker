#ifndef PLAYER_H
#define PLAYER_H

#include <stdbool.h>
#include "cards.h"

#define MAXPLAYERS 8
#define STARTING_CREDITS 100

typedef enum {
    ACTIVE,
    FOLDED,
    NOT_PLAYING
} Status;

typedef struct {
    char name[50];
    int credits;
    Status status;
    char cards[2][3];
    Hand *hand;
    bool dealer;
    int currentBet;
} Player;

void clearInputBuffer();
int setup(Player players[]);
int setupWithAI(Player players[]);

#endif
