#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#ifndef CARDS_H
#define CARDS_H

#define HEARTS 0
#define DIAMONDS 1
#define CLUBS 2
#define SPADES 3

#define ACE 1
#define JACK 11
#define QUEEN 12
#define KING 13

typedef struct card {
    int value;  // 1-13 (Ace=1, Jack=11, Queen=12, King=13)
    int suit;   // 0-3 (Hearts=0, Diamonds=1, Clubs=2, Spades=3)
    struct card *next;
} Card;

typedef struct {
    int size;
    Card *first;
} Hand;

Hand *createHand(void);
Card *createCard(int value, int suit);
void addCard(Hand *h, Card *c);
Card *getTop(Hand *h);
Card *getCard(Hand *h, int pos);
void shuffle(Hand *h);
void printCard(char *str, Card *c);
void printHand(char *str, Hand *h);
Hand *createDeck(int numpacks, int shuffled);
void sortHand(Hand *h);
void freeHand(Hand *h, int delCards);
void cardValString(char s[], int val);

#endif
