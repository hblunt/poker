#ifndef GAME_H
#define GAME_H

#include "player.h"
#include "scoringsystem.h"

#define SMALL_BLIND 5
#define BIG_BLIND 10

void dealHand(Player players[], int numPlayers, Hand *deck, Hand *communityCards);
int playHandAI(int numPlayers, Player players[]);
int playHand(int numPlayers, Player players[]);
bool predictionRound(Player players[], int numPlayers, int *pot, int roundNum, Hand* communityCards, int cardsRevealed, int startPosition, int *currentBetAmount);
int endGame(Player players[], int numPlayers, int pot, Hand* communityCards);
void combineCards(Player *player, Hand *communityCards, Card combined[], int *numCards);
int findNextActivePlayer(Player players[], int numPlayers, int currentPos, int offset);
void resetCurrentBets(Player players[], int numPlayers);
bool allBetsMatched(Player players[], int numPlayers, int currentBet);
void clearScreen();
void pause();

// Training log viewing functions
void viewTrainingLogs();
void showBasicTrainingLog();
void showSelfPlayTrainingLog();

#endif

