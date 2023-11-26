from sklearn.tree import DecisionTreeClassifier
import pickle

# Sample data for demonstration
data = [
    {"name": "Alice", "age": 25, "interests": ["hiking", "reading"], "gender": "female"},
    {"name": "Bob", "age": 30, "interests": ["gaming", "movies"], "gender": "male"},
    # Add more sample data
]

# Features: 'age', 'interests', 'gender'
X = [[user["age"], len(user["interests"]), user["gender"] == "female"] for user in data]
# Labels: 1 if the user is interested, 0 if not (for simplicity)
y = [1, 0]  # Add more labels

# Train a simple decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Save the trained model using pickle
with open("model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)

def find_match(user_data):
    # Prepare input for the trained model
    input_data = [[user_data["age"], len(user_data["interests"]), user_data["gender"] == "female"]]

    # Load the trained model
    with open("model.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)

    # Make a prediction
    prediction = loaded_model.predict(input_data)

    # Return the result and the prediction
    return prediction[0] == 1, prediction[0]

def add_user_to_data(user_data):
    data.append(user_data)

def display_match_details(matched_user):
    print("\nMatch Details:")
    print(f"Name: {matched_user['name']}")
    print(f"Age: {matched_user['age']}")
    print(f"Interests: {', '.join(matched_user['interests'])}")
    print(f"Gender: {matched_user['gender']}")

if __name__ == "__main__":
    print("Simple Dating App")
    print("------------------")

    name = input("Enter your name: ")
    age = int(input("Enter your age: "))
    interests = input("Enter your interests (comma-separated): ").split(",")
    gender = input("Enter your gender (male/female): ")

    user_data = {"name": name, "age": age, "interests": interests, "gender": gender}

    is_match, matched_user = find_match(user_data)

    if is_match:
        print("Congratulations! You have a match!")
        display_match_details(matched_user)
    else:
        print("Sorry, no match found.")
        print("Adding your details to the sample data...")
        add_user_to_data(user_data)

    # Display the updated sample data
    print("\nUpdated Sample Data:")
    print(data)
