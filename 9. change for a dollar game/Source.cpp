/*
create a change-counting game that gets the user to enter the number of coins required 
to make exactuly one dollar. the program should ask the user to enter the number of 
pennies, nickels, dimes, and quarters. if the total value of the coins entered is equal to 
one dollar, the program should congratulate the user for winning the game. otherwise,
the program should display a message intidcating whether the amount entered was more
than or less than one dollar
*/

#include <iostream>
using namespace std;

int main() {

	double penny = 0.01, nickel = 0.05, dime = 0.10, quarter = 0.25;
	double pennies, nickels, dimes, quarters;
	double dollar = 1, givenTotal = 0;
	cout << "please enter the number of quarters used";
	cin >> quarters;
	cout << "please enter the number of dimes";
	cin >> dimes;
	cout << "please enter the number of nickels";
	cin >> nickels;
	cout << "please enter the number of nickels";
	cin >> pennies;
	givenTotal = quarters * quarter + dimes * dime + nickels * nickel + pennies * penny;
	cout << givenTotal;
	if (dollar == givenTotal) 
	{
		cout << "Congrats you entered just a dollar.";
	}
	else if (dollar > givenTotal)
	{
		cout << "You lost by entering a total that was smaller than a doller.";
	}
	else if(dollar < givenTotal)
	{
		cout << "You lost by entering a total that was larger than a dollar.";
	}

	return 0;
}
