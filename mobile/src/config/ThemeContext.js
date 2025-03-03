// ThemeContext.js
import React, { createContext, useState, useEffect } from 'react';
import { Appearance } from 'react-native';

export const ThemeContext = createContext();

const themes = {
  light: {
    primaryColors: {
      inkBlack: '#0F0F0F',
      deepCrimson: '#D72638',
      electricBlue: '#007BFF',
    },
    accentColors: {
      warmBeige: '#F5E6CC',
      graphiteGray: '#3A3A3A',
    },
    typography: {
      primaryFont: 'Montserrat',
      secondaryFont: 'Open Sans',
    },
  },
  dark: {
    primaryColors: {
      inkBlack: '#0F0F0F',
      deepCrimson: '#D72638',
      electricBlue: '#007BFF',
    },
    accentColors: {
      warmBeige: '#F5E6CC',
      graphiteGray: '#3A3A3A',
    },
    typography: {
      primaryFont: 'Montserrat',
      secondaryFont: 'Open Sans',
    },
  },
};

export const ThemeProvider = ({ children }) => {
  const colorScheme = Appearance.getColorScheme();
  const [theme, setTheme] = useState(colorScheme === 'dark' ? themes.dark : themes.light);

  const toggleTheme = () => {
    setTheme(theme === themes.light ? themes.dark : themes.light);
  };

  useEffect(() => {
    const subscription = Appearance.addChangeListener(({ colorScheme }) => {
      setTheme(colorScheme === 'dark' ? themes.dark : themes.light);
    });
    return () => subscription.remove();
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};