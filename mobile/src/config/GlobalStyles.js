// GlobalStyles.js
import { StyleSheet } from 'react-native';
import { ThemeContext } from './ThemeContext';

export const GlobalStyles = (theme) => StyleSheet.create({
  buttonPrimary: {
    backgroundColor: theme.primaryColors.deepCrimson,
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: 5,
  },
  buttonSecondary: {
    backgroundColor: 'transparent',
    borderColor: theme.primaryColors.electricBlue,
    borderWidth: 2,
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: 5,
  },
  buttonTertiary: {
    backgroundColor: 'transparent',
    padding: 12,
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: 5,
  },
  // Add more global styles as needed
});