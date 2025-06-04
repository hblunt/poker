#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "scoringsystem.h"

// Find the best 5-card hand from 7 cards
HandScore findBestHand(Card cards[], int numCards) {
    if (numCards < 5) {
        HandScore invalidScore = {0};
        return invalidScore;
    }

    // Check all possible 21 5-card combinations
    HandScore bestScore = {0};
    Card combination[5];

    // Indices for the cards in the combination
    int indices[5] = {0, 1, 2, 3, 4};
    bool done = false;

    while (!done) {
        // Build the current combination
        for (int i = 0; i < 5; i++) {
            combination[i] = cards[indices[i]];
        }

        // Score this combination
        HandScore currentScore = scoreHand(combination, 5);

        // Update best score
        if (compareHandScores(currentScore, bestScore) > 0 || bestScore.rank == 0) {
            bestScore = currentScore;
        }

        // Generate the next combination
        int i = 4;
        while (i >= 0 && indices[i] == numCards - 5 + i) {
            i--;
        }

        if (i < 0) {
            // No more combinations
            done = true;
        } else {
            // Increment the index and reset all following indices
            indices[i]++;
            for (int j = i + 1; j < 5; j++) {
                indices[j] = indices[j-1] + 1;
            }
        }
    }

    return bestScore;
}

// Function to calculate the score for a 5-card hand
HandScore scoreHand(Card hand[], int size) {
    HandScore score = {0};
    int values[14] = {0}; // Count of each card value (index 0 not used)
    int suits[4] = {0};   // Count of each suit
    int i, highestCard = 0;

    // Count occurrences of each value and suit
    for (i = 0; i < size; i++) {
        values[hand[i].value]++;
        suits[hand[i].suit]++;
        if (hand[i].value > highestCard) {
            highestCard = hand[i].value;
        }
    }

    // Check for flush (all cards same suit)
    bool isFlush = false;
    for (i = 0; i < 4; i++) {
        if (suits[i] == 5) {
            isFlush = true;
            break;
        }
    }

    // Check for straight (5 consecutive values)
    bool isStraight = false;
    int straightHighCard = 0;

    // Special case for A-5 straight (Ace counts as 1)
    if (values[1] && values[2] && values[3] && values[4] && values[5]) {
        isStraight = true;
        straightHighCard = 5;
    }

    // Normal straights
    for (i = 1; i <= 9; i++) {
        if (values[i] && values[i+1] && values[i+2] && values[i+3] && values[i+4]) {
            isStraight = true;
            straightHighCard = i + 4;
        }
    }

    // Special case for 10-J-Q-K-A straight
    if (values[1] && values[10] && values[11] && values[12] && values[13]) {
        isStraight = true;
        straightHighCard = 14; // Ace high
    }

    // Determine hand rank and values
    if (isFlush && isStraight) {
        if (straightHighCard == 14) {
            score.rank = ROYAL_FLUSH;
        } else {
            score.rank = STRAIGHT_FLUSH;
            score.primaryValue = straightHighCard;
        }
    } else if (isFlush) {
        score.rank = FLUSH;
        // Primary value is the highest card
        for (i = 13; i >= 1; i--) {
            if (values[i]) {
                score.primaryValue = i;
                break;
            }
        }
    } else if (isStraight) {
        score.rank = STRAIGHT;
        score.primaryValue = straightHighCard;
    } else {
        // Check for four of a kind, full house, etc.
        int pairCount = 0;
        int threeOfAKind = 0;
        int fourOfAKind = 0;

        for (i = 1; i <= 13; i++) {
            if (values[i] == 4) {
                fourOfAKind = i;
            } else if (values[i] == 3) {
                threeOfAKind = i;
            } else if (values[i] == 2) {
                pairCount++;
                if (pairCount == 1) {
                    score.secondaryValue = score.primaryValue;
                    score.primaryValue = i;
                } else if (pairCount == 2 && i > score.primaryValue) {
                    score.secondaryValue = score.primaryValue;
                    score.primaryValue = i;
                } else if (pairCount == 2) {
                    score.secondaryValue = i;
                }
            }
        }

        if (fourOfAKind) {
            score.rank = FOUR_OF_A_KIND;
            score.primaryValue = fourOfAKind;
        } else if (threeOfAKind && pairCount >= 1) {
            score.rank = FULL_HOUSE;
            score.primaryValue = threeOfAKind;
            // Secondary value is the pair
            for (i = 13; i >= 1; i--) {
                if (values[i] == 2) {
                    score.secondaryValue = i;
                    break;
                }
            }
        } else if (threeOfAKind) {
            score.rank = THREE_OF_A_KIND;
            score.primaryValue = threeOfAKind;
        } else if (pairCount == 2) {
            score.rank = TWO_PAIR;
            // primaryValue and secondaryValue already set
        } else if (pairCount == 1) {
            score.rank = PAIR;
            // primaryValue already set
        } else {
            score.rank = HIGH_CARD;
            score.primaryValue = highestCard;
        }
    }

    // Store high cards for tie-breaking
    int cardIdx = 0;
    for (i = 13; i >= 1 && cardIdx < 5; i--) {
        if (values[i] == 1) {
            score.highCards[cardIdx++] = i;
        }
    }

    return score;
}

// Convert hand score to a numeric value for comparison
long long handScoreToNumeric(HandScore score) {
    // Use shifts to create a hierarchical score
    // Format: [rank][primaryValue][secondaryValue][highCards]
    long long numericScore = 0;

    numericScore += (long long)score.rank * 10000000000; // 10^10
    numericScore += (long long)score.primaryValue * 100000000; // 10^8
    numericScore += (long long)score.secondaryValue * 1000000; // 10^6

    // Add high cards with decreasing weight
    for (int i = 0; i < 5; i++) {
        numericScore += (long long)score.highCards[i] * pow(100, 4-i);
    }

    return numericScore;
}

// Compare two hand scores, return 1 if a > b, -1 if a < b, 0 if equal
int compareHandScores(HandScore a, HandScore b) {
    // Compare hand ranks first
    if (a.rank > b.rank) return 1;
    if (a.rank < b.rank) return -1;

    // Same rank, compare primary values
    if (a.primaryValue > b.primaryValue) return 1;
    if (a.primaryValue < b.primaryValue) return -1;

    // Same primary value, compare secondary values
    if (a.secondaryValue > b.secondaryValue) return 1;
    if (a.secondaryValue < b.secondaryValue) return -1;

    // Same secondary value, compare high cards
    for (int i = 0; i < 5; i++) {
        if (a.highCards[i] > b.highCards[i]) return 1;
        if (a.highCards[i] < b.highCards[i]) return -1;
    }

    // Equal hands
    return 0;
}

// Get string representation of a hand rank
void handRankToString(int rank, char *str) {
    switch (rank) {
        case HIGH_CARD:
            strcpy(str, "High Card");
            break;
        case PAIR:
            strcpy(str, "Pair");
            break;
        case TWO_PAIR:
            strcpy(str, "Two Pair");
            break;
        case THREE_OF_A_KIND:
            strcpy(str, "Three of a Kind");
            break;
        case STRAIGHT:
            strcpy(str, "Straight");
            break;
        case FLUSH:
            strcpy(str, "Flush");
            break;
        case FULL_HOUSE:
            strcpy(str, "Full House");
            break;
        case FOUR_OF_A_KIND:
            strcpy(str, "Four of a Kind");
            break;
        case STRAIGHT_FLUSH:
            strcpy(str, "Straight Flush");
            break;
        case ROYAL_FLUSH:
            strcpy(str, "Royal Flush");
            break;
        default:
            strcpy(str, "Unknown Hand");
    }
}

// Get string representation of card value
void valueToString(int value, char *str) {
    switch (value) {
        case 1:
            strcpy(str, "Ace");
            break;
        case 11:
            strcpy(str, "Jack");
            break;
        case 12:
            strcpy(str, "Queen");
            break;
        case 13:
            strcpy(str, "King");
            break;
        default:
            sprintf(str, "%d", value);
    }
}

void describeHandScore(HandScore score, char *description) {
    char rankStr[20];
    char valueStr[10];
    char secondaryStr[10];

    handRankToString(score.rank, rankStr);
    valueToString(score.primaryValue, valueStr);

    switch (score.rank) {
        case HIGH_CARD:
            sprintf(description, "%s: %s high", rankStr, valueStr);
            break;
        case PAIR:
            sprintf(description, "%s of %ss", rankStr, valueStr);
            break;
        case TWO_PAIR:
            valueToString(score.secondaryValue, secondaryStr);
            sprintf(description, "%s: %ss and %ss", rankStr, valueStr, secondaryStr);
            break;
        case THREE_OF_A_KIND:
            sprintf(description, "%s: Three %ss", rankStr, valueStr);
            break;
        case STRAIGHT:
            sprintf(description, "%s to %s", rankStr, valueStr);
            break;
        case FLUSH:
            sprintf(description, "%s with %s high", rankStr, valueStr);
            break;
        case FULL_HOUSE:
            valueToString(score.secondaryValue, secondaryStr);
            sprintf(description, "%s: %ss full of %ss", rankStr, valueStr, secondaryStr);
            break;
        case FOUR_OF_A_KIND:
            sprintf(description, "%s: Four %ss", rankStr, valueStr);
            break;
        case STRAIGHT_FLUSH:
            sprintf(description, "%s to %s", rankStr, valueStr);
            break;
        case ROYAL_FLUSH:
            strcpy(description, rankStr);
            break;
        default:
            strcpy(description, "Unknown Hand");
    }
}

