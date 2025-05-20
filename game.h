#ifndef GAME_H
#define GAME_H

#include "player.h"
#include "scoringsystem.h"

void dealHand(Player players[], int numPlayers, Hand *deck, Hand *communityCards);
int playHand(int numPlayers, Player players[]);
bool predictionRound(Player players[], int numPlayers, int *pot, int roundNum, Hand* communityCards, int cardsRevealed);
int endGame(Player players[], int numPlayers, int pot, Hand* communityCards);
void combineCards(Player *player, Hand *communityCards, Card combined[], int *numCards);

#endif

