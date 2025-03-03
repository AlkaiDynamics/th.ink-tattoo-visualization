// HomeScreen.js
import React, { useContext } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import Video from 'react-native-video';
import { ThemeContext } from '../config/ThemeContext';
import { GlobalStyles } from '../config/GlobalStyles';
import SubscriptionBanner from '../components/SubscriptionBanner'; // Importing SubscriptionBanner

const HomeScreen = ({ navigation }) => {
  const { theme } = useContext(ThemeContext);
  const styles = createStyles(theme);

  return (
    <View style={styles.container}>
      {/* Subscription Banner */}
      <SubscriptionBanner navigation={navigation} />

      {/* Background Video */}
      <Video
        source={require('../../assets/background.mp4')} // Ensure this path is correct
        style={styles.backgroundVideo}
        repeat
        muted
        resizeMode="cover"
      />
      
      {/* Title */}
      <Text style={styles.title}>Th.ink Tattoo Visualization</Text>
      
      {/* CTA Buttons */}
      <TouchableOpacity
        style={GlobalStyles(theme).buttonPrimary}
        onPress={() => navigation.navigate('ARViewer')}
      >
        <Text style={styles.buttonText}>AR Viewer</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={GlobalStyles(theme).buttonSecondary}
        onPress={() => navigation.navigate('TattooGenerator')}
      >
        <Text style={styles.buttonText}>Tattoo Generator</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={GlobalStyles(theme).buttonTertiary}
        onPress={() => navigation.navigate('Marketplace')}
      >
        <Text style={styles.buttonText}>Marketplace</Text>
      </TouchableOpacity>
      {/* Add more buttons as needed */}
    </View>
  );
};

const createStyles = (theme) =>
  StyleSheet.create({
    container: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      padding: 16,
    },
    backgroundVideo: {
      position: 'absolute',
      top: 0,
      left: 0,
      bottom: 0,
      right: 0,
    },
    title: {
      fontSize: 24,
      fontWeight: 'bold',
      marginBottom: 20,
      color: theme.primaryColors.inkBlack,
      fontFamily: theme.typography.primaryFont,
      textAlign: 'center',
    },
    buttonText: {
      color: '#FFFFFF',
      fontFamily: theme.typography.secondaryFont,
      fontSize: 16,
    },
  });

export default HomeScreen;