// AppNavigator.js
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { ThemeProvider, ThemeContext } from '../config/ThemeContext';
import { GlobalStyles } from '../config/GlobalStyles';
import HomeScreen from '../screens/HomeScreen';
import MarketplaceScreen from '../screens/MarketplaceScreen';
import SavedTattoosScreen from '../screens/SavedTattoosScreen';
import ProfileScreen from '../screens/ProfileScreen';

const Tab = createBottomTabNavigator();

const AppNavigator = () => {
  return (
    <ThemeProvider>
      <ThemeContext.Consumer>
        {({ theme }) => (
          <NavigationContainer>
            <Tab.Navigator
              screenOptions={{
                headerStyle: {
                  backgroundColor: theme.primaryColors.deepCrimson,
                },
                headerTintColor: theme.accentColors.graphiteGray,
                tabBarStyle: {
                  backgroundColor: theme.primaryColors.inkBlack,
                },
                tabBarActiveTintColor: theme.primaryColors.deepCrimson,
                tabBarInactiveTintColor: theme.accentColors.graphiteGray,
              }}
            >
              <Tab.Screen name="Home" component={HomeScreen} />
              <Tab.Screen name="Marketplace" component={MarketplaceScreen} />
              <Tab.Screen name="Saved Tattoos" component={SavedTattoosScreen} />
              <Tab.Screen name="Profile" component={ProfileScreen} />
            </Tab.Navigator>
          </NavigationContainer>
        )}
      </ThemeContext.Consumer>
    </ThemeProvider>
  );
};

export default AppNavigator;