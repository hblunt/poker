#include <stdio.h>

#ifndef SCORINGSYSTEM_H
#define SCORINGSYSTEM_H

#include "cards.h"

#define HIGH_CARD 1
#define PAIR 2
#define TWO_PAIR 3
#define THREE_OF_A_KIND 4
#define STRAIGHT 5
#define FLUSH 6
#define FULL_HOUSE 7
#define FOUR_OF_A_KIND 8
#define STRAIGHT_FLUSH 9
#define ROYAL_FLUSH 10

typedef struct {
    int rank;           // The rank of the hand (pair, flush, etc.)
    int primaryValue;   // Primary value for the hand (e.g., value of a pair)
    int secondaryValue; // Secondary value (e.g., value of the second pair in two pair)
    int highCards[5];   // Values of high cards in descending order
} HandScore;

// Hand scoring functions
HandScore scoreHand(Card hand[], int size);
HandScore findBestHand(Card cards[], int numCards);
long long handScoreToNumeric(HandScore score);
int compareHandScores(HandScore a, HandScore b);
void handRankToString(int rank, char *str);
void valueToString(int value, char *str);
void describeHandScore(HandScore score, char *description);

#endif

