"""
SESSION 5 - TASK 2: Zomato-style Restaurant Info Q&A Chatbot
============================================================
A simple interactive and rule/knowledge-based Q&A assistant
that answers customer queries regarding restaurant cuisines,
operating status, ratings, and recommendations.
"""

class ZomatoInfoChatbot:
    def __init__(self):
        # Database of popular restaurants and cuisines
        self.restaurant_database = {
            "pizza palace": {"cuisine": "Italian", "rating": 4.6, "city": "Mumbai", "popular_dishes": "Wood-fired Margherita, Truffle Pasta"},
            "biryani central": {"cuisine": "Hyderabadi / Indian", "rating": 4.8, "city": "Hyderabad", "popular_dishes": "Kacche Gosht Ki Biryani, Mirchi Ka Salan"},
            "taco fiesta": {"cuisine": "Mexican", "rating": 4.4, "city": "Bengaluru", "popular_dishes": "Chipotle Chicken Tacos, Churros"},
            "wok & roll": {"cuisine": "Chinese / Pan-Asian", "rating": 4.5, "city": "Delhi", "popular_dishes": "Dim Sums, Hakka Noodles"},
            "swagath": {"cuisine": "Coastal & South Indian", "rating": 4.7, "city": "Chennai", "popular_dishes": "Ghee Roast Dosa, Fish Curry"},
            "rajdhani thali": {"cuisine": "Gujarati & Rajasthani", "rating": 4.9, "city": "Ahmedabad", "popular_dishes": "Dal Baati Churma, Dhokla, Aamras"}
        }

    def answer_query(self, restaurant_name: str, cuisine: str = None) -> str:
        """
        Given a restaurant name and optionally a cuisine, returns an informative confirmation.
        """
        key = restaurant_name.strip().lower()
        
        if key in self.restaurant_database:
            info = self.restaurant_database[key]
            known_cuisine = info["cuisine"]
            
            if cuisine:
                # Check if queried cuisine matches
                if cuisine.strip().lower() in known_cuisine.lower():
                    return f"Yes! {restaurant_name.title()} serves authentic {known_cuisine} cuisine with a {info['rating']}/5 rating. Popular dishes: {info['popular_dishes']}."
                else:
                    return f"No, {restaurant_name.title()} specializes in {known_cuisine} cuisine rather than {cuisine.title()}."
            else:
                return f"Yes, {restaurant_name.title()} serves {known_cuisine} cuisine (Rated {info['rating']}/5.0). Famous for: {info['popular_dishes']}."
        else:
            if cuisine:
                return f"We found multiple spots for {cuisine.title()}! Would you like recommendations similar to '{restaurant_name.title()}'?"
            return f"Sorry, '{restaurant_name}' was not found in our featured catalog. Please check the spelling or search by cuisine."


def run_chatbot_demo():
    bot = ZomatoInfoChatbot()
    print("=" * 65)
    print("ZOMATO RESTAURANT Q&A CHATBOT DEMO")
    print("=" * 65)
    
    test_queries = [
        ("Pizza Palace", "Italian"),
        ("Biryani Central", "Hyderabadi"),
        ("Pizza Palace", "Chinese"),
        ("Rajdhani Thali", "Gujarati"),
        ("Wok & Roll", None),
        ("Burger King", "American")
    ]
    
    for rest, cuisine in test_queries:
        if cuisine:
            print(f"User Query: Does '{rest}' serve {cuisine}?")
        else:
            print(f"User Query: What does '{rest}' serve?")
        response = bot.answer_query(rest, cuisine)
        print(f"Bot Response: {response}\n")


if __name__ == "__main__":
    run_chatbot_demo()
