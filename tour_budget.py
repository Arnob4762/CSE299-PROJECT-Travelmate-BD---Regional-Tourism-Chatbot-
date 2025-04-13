#!/usr/bin/env python3
"""
Tour Budget Calculator
This module allows the user to choose a destination, transportation mode, hotel category, and restaurant category,
enter trip details (days, nights, season), and then calculates the estimated total budget with a detailed memo.
All data is hardcoded based on the provided travel budget information.
"""

import streamlit as st

def get_average_cost(cost_range):
    """Return the average of the given cost range dictionary (with keys 'min' and 'max')."""
    return (cost_range["min"] + cost_range["max"]) / 2

def show_budget_calculator():
    st.subheader("Tour Budget Calculator")
    
    # Hardcoded destination data
    destinations = {
        "Cox's Bazar": {
            "hotels": {
                "Luxury": {
                    "min": 8000,
                    "max": 15000,
                    "examples": ["Hotel The Cox Today", "Long Beach Hotel", "Sea Pearl Beach Resort & Spa"]
                },
                "Mid-Range": {
                    "min": 2500,
                    "max": 5000,
                    "examples": ["Hotel Coastal Peace", "Best Western Heritage Hotel"]
                },
                "Budget": {
                    "min": 1000,
                    "max": 2000,
                    "examples": ["Hotel Beach Way", "Hotel Media"]
                }
            },
            "restaurants": {
                "Budget": {
                    "min": 200,
                    "max": 400,
                    "examples": ["Poushee Restaurant"]
                },
                "Mid-Range": {
                    "min": 500,
                    "max": 1500,
                    "examples": ["Jhawban Restaurant"]
                },
                "Luxury": {
                    "min": 2000,
                    "max": 4000,
                    "examples": ["Luxury Dining Option"]
                }
            },
            "transport": {
                "Bus": {"min": 500, "max": 1500},
                "Private Car": {"min": 10000, "max": 18000},
                "Flight": {"min": 4000, "max": 7000}
            }
        },
        "Saint Martin's Island": {
            "hotels": {
                "Luxury": {
                    "min": 4000,
                    "max": 9000,
                    "examples": ["Saint Martin’s Resort", "Paradise Beach Resort"]
                },
                "Budget": {
                    "min": 800,
                    "max": 2500,
                    "examples": ["Neelgiri Guesthouse", "Saint Martin’s Guesthouse"]
                },
                "Camping": {
                    "min": 500,
                    "max": 1000,
                    "examples": ["Camping"]
                }
            },
            "restaurants": {
                "Budget": {
                    "min": 500,
                    "max": 1500,
                    "examples": ["Saint Martin’s Resort Restaurant"]
                },
                "Mid-Range": {
                    "min": 1500,
                    "max": 3000,
                    "examples": ["Monir Restaurant"]
                }
            },
            "transport": {
                "Bus": {"min": 1000, "max": 2000},
                "Private Car": {"min": 15000, "max": 20000},
                "Flight + Bus": {"min": 5500, "max": 9000},
                "Ferry": {"min": 500, "max": 1200}
            }
        },
        "Bandarban": {
            "hotels": {
                "Luxury": {
                    "min": 8000,
                    "max": 15000,
                    "examples": ["Chimbuk Hill Resort", "Shangri-La Resort"]
                },
                "Mid-Range": {
                    "min": 3500,
                    "max": 6000,
                    "examples": ["Hotel Hill Town", "Zakir Resort"]
                },
                "Budget": {
                    "min": 1500,
                    "max": 3000,
                    "examples": ["The Green Hill Hotel", "Ruma Eco Resort"]
                }
            },
            "restaurants": {
                "Budget": {
                    "min": 200,
                    "max": 500,
                    "examples": ["Bandarban Special Foods"]
                },
                "Mid-Range": {
                    "min": 500,
                    "max": 1000,
                    "examples": ["Bandarban Special Foods"]
                }
            },
            "transport": {
                "Bus": {"min": 500, "max": 1000},
                "Private Car": {"min": 10000, "max": 15000},
                "Flight + Car": {"min": 4000, "max": 8000}
            }
        },
        "Sundarbans": {
            "hotels": {
                "Luxury": {
                    "min": 5000,
                    "max": 10000,
                    "examples": ["Sundarban Tiger Roar Resort", "Sundarban Riverside Holiday Resort"]
                },
                "Mid-Range": {
                    "min": 2000,
                    "max": 4000,
                    "examples": ["Sundarban Jungle Camp"]
                },
                "Budget": {
                    "min": 500,
                    "max": 1500,
                    "examples": []
                }
            },
            "restaurants": {
                "Budget": {
                    "min": 300,
                    "max": 700,
                    "examples": ["Sundarbans Special Foods"]
                },
                "Mid-Range": {
                    "min": 700,
                    "max": 1500,
                    "examples": ["Sundarbans Special Foods"]
                }
            },
            "transport": {
                "Bus": {"min": 600, "max": 1800},
                "Train": {"min": 450, "max": 1800},
                "Flight + Bus": {"min": 3500, "max": 6500}
            }
        },
        "Rangamati": {
            "hotels": {
                "Luxury": {
                    "min": 6000,
                    "max": 12000,
                    "examples": ["Hotel Sufia International", "Parjatan Holiday Complex"]
                },
                "Mid-Range": {
                    "min": 3000,
                    "max": 6000,
                    "examples": ["Hotel Nadisa International", "Aronnok Holiday Resort"]
                },
                "Budget": {
                    "min": 1500,
                    "max": 3000,
                    "examples": ["Hotel Green Castle", "Lake View Resort"]
                }
            },
            "restaurants": {
                "Budget": {
                    "min": 200,
                    "max": 500,
                    "examples": ["Rangamati Special Foods"]
                },
                "Mid-Range": {
                    "min": 500,
                    "max": 1200,
                    "examples": ["Rangamati Special Foods"]
                }
            },
            "transport": {
                "Bus": {"min": 800, "max": 1500},
                "Private Car": {"min": 10000, "max": 18000},
                "Flight + Car": {"min": 5000, "max": 9000}
            }
        },
        "Khagrachari & Sajek Valley": {
            "hotels": {
                "Luxury": {
                    "min": 8000,
                    "max": 12000,
                    "examples": ["Sajek Resort", "Megh Machang Resort"]
                },
                "Mid-Range": {
                    "min": 4000,
                    "max": 8000,
                    "examples": ["Runmoy Resort", "Hilltop Inn"]
                },
                "Budget": {
                    "min": 1500,
                    "max": 3000,
                    "examples": ["Sajek Eco Resort", "Konglak Hill View Cottage"]
                }
            },
            "restaurants": {
                "Budget": {
                    "min": 200,
                    "max": 500,
                    "examples": ["Khagrachari/Sajek Special Foods"]
                },
                "Mid-Range": {
                    "min": 500,
                    "max": 1200,
                    "examples": ["Khagrachari/Sajek Special Foods"]
                }
            },
            "transport": {
                "Bus": {"min": 900, "max": 1500},
                "Private Car": {"min": 12000, "max": 20000},
                "Flight + Car": {"min": 5000, "max": 9000}
            }
        },
        "Sylhet": {
            "hotels": {
                "Luxury": {
                    "min": 6500,
                    "max": 15000,
                    "examples": ["Nirvana Inn", "Richmond Hotel & Suites", "Shuktara Hotel & Resort"]
                },
                "Mid-Range": {
                    "min": 2500,
                    "max": 6000,
                    "examples": ["Hotel Grand Selim", "Hotel Purbani International"]
                },
                "Budget": {
                    "min": 800,
                    "max": 2000,
                    "examples": ["Local guesthouses", "Budget hotels in Zindabazar"]
                }
            },
            "restaurants": {
                "Budget": {
                    "min": 50,
                    "max": 150,
                    "examples": ["Sylhet Special Foods"]
                },
                "Mid-Range": {
                    "min": 250,
                    "max": 800,
                    "examples": ["Sylhet Special Foods"]
                },
                "Luxury": {
                    "min": 1000,
                    "max": 2500,
                    "examples": ["Sylhet Special Foods"]
                }
            },
            "transport": {
                "Bus": {"min": 400, "max": 1500},
                "Train": {"min": 300, "max": 1200},
                "Private Car": {"min": 6000, "max": 10000},
                "Flight": {"min": 3000, "max": 5000}
            }
        },
        "Kuakata": {
            "hotels": {
                "Luxury": {
                    "min": 5000,
                    "max": 10000,
                    "examples": ["Kuakata Beach Resort", "Hotel Sea World"]
                },
                "Mid-Range": {
                    "min": 2500,
                    "max": 5000,
                    "examples": ["Rakhine Hotel & Resort", "Fatrar Island Resort"]
                },
                "Budget": {
                    "min": 1000,
                    "max": 2000,
                    "examples": ["Hotel Motijheel", "Budget guesthouses"]
                }
            },
            "restaurants": {
                "Budget": {
                    "min": 200,
                    "max": 500,
                    "examples": ["Kuakata Special Foods"]
                },
                "Mid-Range": {
                    "min": 500,
                    "max": 1000,
                    "examples": ["Kuakata Special Foods"]
                },
                "Luxury": {
                    "min": 1000,
                    "max": 1500,
                    "examples": ["Kuakata Special Foods"]
                }
            },
            "transport": {
                "Bus": {"min": 400, "max": 1500},
                "Train + Bus": {"min": 400, "max": 800},
                "Private Car": {"min": 6000, "max": 10000},
                "Flight + Bus": {"min": 3200, "max": 5400}
            }
        }
    }
    
    # Season multipliers
    season_multiplier = {
        "Off-Season": 0.9,
        "On-Season": 1.2
    }
    
    # Input Section using Streamlit widgets
    destination_list = list(destinations.keys())
    destination = st.selectbox("Select Destination:", destination_list)
    
    trans_options = list(destinations[destination]["transport"].keys())
    transport_selected = st.selectbox(f"Select Transportation Option for {destination}:", trans_options)
    
    hotel_options = list(destinations[destination]["hotels"].keys())
    hotel_selected = st.selectbox(f"Select Hotel Category for {destination}:", hotel_options)
    
    rest_options = list(destinations[destination]["restaurants"].keys())
    restaurant_selected = st.selectbox(f"Select Restaurant Category for {destination}:", rest_options)
    
    days = st.number_input("Enter the number of days for your trip:", min_value=1, step=1, value=1)
    nights = st.number_input("Enter the number of hotel nights:", min_value=1, step=1, value=1)
    meals_per_day = 3  # Assuming 3 meals per day
    
    season_options = list(season_multiplier.keys())
    season_selected = st.selectbox("Select Season:", season_options)
    season_factor = season_multiplier[season_selected]
    
    # Calculation Section
    trans_range = destinations[destination]["transport"][transport_selected]
    transport_cost = get_average_cost(trans_range)
    
    hotel_range = destinations[destination]["hotels"][hotel_selected]
    hotel_cost_per_night = get_average_cost(hotel_range)
    hotel_cost = hotel_cost_per_night * nights
    
    rest_range = destinations[destination]["restaurants"][restaurant_selected]
    restaurant_cost_per_meal = get_average_cost(rest_range)
    meal_cost = restaurant_cost_per_meal * meals_per_day * days
    
    base_total = transport_cost + hotel_cost + meal_cost
    total_cost = base_total * season_factor
    
    memo = f"""
    ========== TOUR BUDGET SUMMARY ==========
    Destination: {destination}
    Season: {season_selected} (Multiplier: {season_factor})
    Trip Duration: {days} days, {nights} hotel nights

    **Transportation:** {transport_selected}
      - Average Cost: BDT {transport_cost:.2f}

    **Hotel:** {hotel_selected}
      - Average Cost per Night: BDT {hotel_cost_per_night:.2f}
      - Total Hotel Cost for {nights} nights: BDT {hotel_cost:.2f}
      - Examples: {', '.join(hotel_range['examples'])}

    **Restaurant:** {restaurant_selected}
      - Average Cost per Meal: BDT {restaurant_cost_per_meal:.2f}
      - Total Meal Cost for {days} days (3 meals/day): BDT {meal_cost:.2f}
      - Examples: {', '.join(rest_range['examples'])}

    **Base Total Cost:** BDT {base_total:.2f}
    **Total Estimated Budget (after season adjustment):** BDT {total_cost:.2f}
    ===========================================
    """
    
    st.markdown(memo)
    
if __name__ == "__main__":
    show_budget_calculator()
