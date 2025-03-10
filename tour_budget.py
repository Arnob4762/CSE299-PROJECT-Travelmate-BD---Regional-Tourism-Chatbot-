import streamlit as st

def calculate_tour_budget(hotel_cost_per_night, num_nights, transportation_cost, restaurant_cost_per_day, num_days, attractions_cost=0):
    """
    Calculate the estimated tour budget.
    
    Parameters:
      - hotel_cost_per_night: cost per night for the chosen hotel
      - num_nights: number of nights of stay
      - transportation_cost: cost for transportation
      - restaurant_cost_per_day: average daily restaurant expense
      - num_days: number of days of the trip
      - attractions_cost: total cost for attractions (optional)
      
    Returns:
      - total estimated budget
    """
    total_hotel = hotel_cost_per_night * num_nights
    total_restaurant = restaurant_cost_per_day * num_days
    total_budget = total_hotel + transportation_cost + total_restaurant + attractions_cost
    return total_budget

def show_budget_calculator():
    st.title("Tour Budget Calculator")
    st.markdown("Estimate your tour budget based on options extracted from your documents or sample data.")
    
    # Simulated options (replace with extracted data if available)
    st.markdown("### Destination")
    places = ["Cox's Bazar", "St. Martin's Island", "Bandarban", "Sylhet"]
    selected_place = st.selectbox("Select a destination", places)
    
    st.markdown("### Transportation Options")
    transport_options = {
        "Bus": 500,
        "Train": 700,
        "Car": 1500,
        "Flight": 5000
    }
    selected_transport = st.selectbox("Select a transportation mode", list(transport_options.keys()))
    transportation_cost = transport_options[selected_transport]
    
    st.markdown("### Hotel Options")
    hotels = {
        "Hotel A": 2000,
        "Hotel B": 2500,
        "Hotel C": 3000
    }
    selected_hotel = st.selectbox("Select a hotel", list(hotels.keys()))
    hotel_cost = hotels[selected_hotel]
    
    st.markdown("### Attractions (Optional)")
    attractions_cost = st.number_input("Enter total cost for attractions (BDT):", min_value=0, value=0)
    
    st.markdown("### Restaurant Expenses")
    restaurant_cost = st.number_input("Enter average restaurant expense per day (BDT):", min_value=0, value=0)
    
    st.markdown("### Duration")
    num_nights = st.number_input("Enter number of nights:", min_value=1, value=1)
    num_days = st.number_input("Enter number of days:", min_value=1, value=1)
    
    if st.button("Calculate Budget"):
        total = calculate_tour_budget(hotel_cost, num_nights, transportation_cost, restaurant_cost, num_days, attractions_cost)
        st.success(f"Total estimated tour budget for {selected_place}: {total} BDT")
