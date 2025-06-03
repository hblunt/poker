#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cards.h"

// Create a new empty hand
Hand *createHand(void) {
    Hand *h = malloc(sizeof(Hand));
    if (h == NULL) {
        fprintf(stderr, "Memory allocation failed for hand\n");
        exit(EXIT_FAILURE);
    }
    h->size = 0;
    h->first = NULL;
    return h;
}

// Create a card with specified value and suit
Card *createCard(int value, int suit) {
    Card *c = malloc(sizeof(Card));
    if (c == NULL) {
        fprintf(stderr, "Memory allocation failed for card\n");
        exit(EXIT_FAILURE);
    }
    c->value = value;
    c->suit = suit;
    c->next = NULL;
    return c;
}

// Add a card to a hand (at the beginning for simplicity)
void addCard(Hand *h, Card *c) {
    if (h == NULL || c == NULL) return;

    c->next = h->first;
    h->first = c;
    h->size++;
}

// Get and remove the top card from a hand
Card *getTop(Hand *h) {
    if (h == NULL || h->first == NULL) return NULL;

    Card *c = h->first;
    h->first = c->next;
    c->next = NULL;
    h->size--;

    return c;
}

// Get a card from a hand without removing it
Card *getCard(Hand *h, int pos) {
    if (h == NULL || h->first == NULL || pos < 0 || pos >= h->size) return NULL;

    Card *c = h->first;
    for (int i = 0; i < pos; i++) {
        if (c == NULL) return NULL;
        c = c->next;
    }

    return c;
}

// Shuffle the cards in a hand
void shuffle(Hand *h) {
    if (h == NULL || h->size <= 1) return;

    // Convert linked list to array for easier shuffling
    Card **cards = malloc(h->size * sizeof(Card*));
    if (cards == NULL) {
        fprintf(stderr, "Memory allocation failed during shuffle\n");
        return;
    }

    Card *current = h->first;
    for (int i = 0; i < h->size; i++) {
        cards[i] = current;
        current = current->next;
    }

    for (int i = h->size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        Card *temp = cards[i];
        cards[i] = cards[j];
        cards[j] = temp;
    }

    // Rebuild linked list
    h->first = cards[0];
    for (int i = 0; i < h->size - 1; i++) {
        cards[i]->next = cards[i + 1];
    }
    cards[h->size - 1]->next = NULL;

    free(cards);
}

// Convert card value to string representation
void cardValString(char s[], int val) {
    switch (val) {
        case ACE:
            strcpy(s, "A");
            break;
        case 10:
            strcpy(s, "T");
            break;
        case JACK:
            strcpy(s, "J");
            break;
        case QUEEN:
            strcpy(s, "Q");
            break;
        case KING:
            strcpy(s, "K");
            break;
        default:
            sprintf(s, "%d", val);
            break;
    }
}

// Print a card to a string
void printCard(char *str, Card *c) {
    if (c == NULL) {
        strcpy(str, "??");
        return;
    }

    char val[3];
    cardValString(val, c->value);

    char suit;
    switch (c->suit) {
        case HEARTS:
            suit = 'H';
            break;
        case DIAMONDS:
            suit = 'D';
            break;
        case CLUBS:
            suit = 'C';
            break;
        case SPADES:
            suit = 'S';
            break;
        default:
            suit = '?';
            break;
    }

    sprintf(str, "%s%c", val, suit);
}

// Print all cards in a hand to a string
void printHand(char *str, Hand *h) {
    if (h == NULL || h->first == NULL) {
        strcpy(str, "Empty hand");
        return;
    }

    str[0] = '\0';
    Card *current = h->first;
    while (current != NULL) {
        char cardStr[4];
        printCard(cardStr, current);
        strcat(str, cardStr);
        strcat(str, " ");
        current = current->next;
    }
}

// Create a deck of cards
Hand *createDeck(int numpacks, int shuffled) {
    Hand *deck = createHand();

    for (int pack = 0; pack < numpacks; pack++) {
        for (int suit = 0; suit < 4; suit++) {
            for (int value = 1; value <= 13; value++) {
                Card *c = createCard(value, suit);
                addCard(deck, c);
            }
        }
    }

    if (shuffled) {
        shuffle(deck);
    }

    return deck;
}

// Free a hand and optionally its cards
void freeHand(Hand *h, int delCards) {
    if (h == NULL) return;

    if (delCards) {
        Card *current = h->first;
        while (current != NULL) {
            Card *next = current->next;
            free(current);
            current = next;
        }
    }

    free(h);
}

