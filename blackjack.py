import random

class BlackJackGame():
    def __init__(self):
        card_categories = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        cards_list = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
        card_deck = [(card, category) for category in card_categories for card in cards_list]
        self.deck = card_deck.copy()

    def hand_value(self, hand):
        score = 0
        aces = 0
        for card, _ in hand:
            if card in ["Jack", "Queen", "King"]:
                score += 10
            elif card == "Ace":
                score += 11
                aces += 1
            else:
                score += int(card)
        while score > 21 and aces:
            score -= 10
            aces -= 1

        return score

    def game(self):
        random.shuffle(self.deck)
        player_card = [self.deck.pop(), self.deck.pop()]
        dealer_card = [self.deck.pop(), self.deck.pop()]
        while True:
            player_score = self.hand_value(player_card)
            dealer_score = self.hand_value(dealer_card)
            print("Cards Player Has:", player_card)
            print("Score Of The Player:", player_score)
            print("Dealer's Visible Card:", dealer_card[0])

            choice = input('["hit", "stand"]: ').lower()

            if choice == "hit":
                new_card = self.deck.pop()
                player_card.append(new_card)
                player_score = self.hand_value(player_card)
            elif choice == "stand":
                break
            else:
                print("Invalid choice. Please try again.")
                continue

            if player_score > 21:
                print("Cards Dealer Has:", dealer_card)
                print("Score Of The Dealer:", dealer_score)
                print("Cards Player Has:", player_card)
                print("Score Of The Player:", player_score)
                print("Dealer wins (Player Loss Because Player Score is exceeding 21)")
                break

        while dealer_score < 17:
            new_card = self.deck.pop()
            dealer_card.append(new_card)
            dealer_score = self.hand_value(dealer_card)

        print("Dealer reveals cards:", dealer_card)
        print("Dealer's Score:", dealer_score)
        print("\n")

        if dealer_score > 21:
            print("Player wins (Dealer Loss Because Dealer Score is exceeding 21)")
        elif player_score > dealer_score:
            print("Player wins (Player Has Higher Score than Dealer)")
        elif dealer_score > player_score:
            print("Dealer wins (Dealer Has Higher Score than Player)")
        else:
            print("It's a tie")

        print("Final Cards - Player:", player_card, "Score:", player_score)
        print("Final Cards - Dealer:", dealer_card, "Score:", dealer_score)
