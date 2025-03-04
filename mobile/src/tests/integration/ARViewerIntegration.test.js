import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import ARViewerScreen from '../../screens/ARViewerScreen';
import { ThemeContext } from '../../config/ThemeContext';
import { AuthContext } from '../../config/firebase';

jest.mock('react-native-arkit', () => ({
  ARKit: () => null,
}));

jest.mock('../../config/firebase', () => ({
  AuthContext: {
    signIn: jest.fn(),
    signOut: jest.fn(),
    getCurrentUser: jest.fn(() => ({ uid: 'test_user' })),
  },
}));

describe('ARViewerScreen Integration Tests', () => {
  const theme = {
    primaryColors: { inkBlack: '#000000', electricBlue: '#00FFFF' },
    accentColors: { warmBeige: '#F5F5DC', graphiteGray: '#4B4B4B' },
    typography: { secondaryFont: 'Arial' },
  };

  it('should render ARViewerScreen and handle tattoo application flow', async () => {
    const { getByText, getByTestId } = render(
      <ThemeContext.Provider value={{ theme }}>
        <ARViewerScreen />
      </ThemeContext.Provider>
    );

    const applyAnotherButton = getByText('Apply Another Tattoo');
    const finalizeButton = getByText('Finalize & Save');

    expect(applyAnotherButton).toBeTruthy();
    expect(finalizeButton).toBeTruthy();

    fireEvent.press(applyAnotherButton);
    await waitFor(() => {
      // Verify logic to apply another tattoo (mock API call)
    });

    fireEvent.press(finalizeButton);
    await waitFor(() => {
      // Verify logic to finalize and save tattoo (mock API call)
    });
  });
});