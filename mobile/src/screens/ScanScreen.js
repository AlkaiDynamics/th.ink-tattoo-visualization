// ScanScreen.js
import React, { useContext, useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Image, Animated, Alert } from 'react-native';
import { ThemeContext } from '../config/ThemeContext';
import { GlobalStyles } from '../config/GlobalStyles';
// Import AR library or necessary modules here, e.g., react-native-arkit or react-native-viro

const ScanScreen = ({ navigation }) => {
  const { theme } = useContext(ThemeContext);
  const styles = createStyles(theme);

  const [isScanning, setIsScanning] = useState(false);
  const [scanValid, setScanValid] = useState(false);
  const pulseAnim = useState(new Animated.Value(1))[0];

  useEffect(() => {
    if (isScanning) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.5,
            duration: 1000,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 1000,
            useNativeDriver: true,
          }),
        ])
      ).start();
      // Simulate scanning process
      setTimeout(() => {
        setIsScanning(false);
        setScanValid(true);
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 500,
          useNativeDriver: true,
        }).start();
      }, 5000); // 5 seconds scanning simulation
    }
  }, [isScanning, pulseAnim]);

  const startScan = () => {
    setIsScanning(true);
    setScanValid(false);
    // Initialize scanning logic here
  };

  const confirmScan = () => {
    Alert.alert('Scan Confirmed', 'Your scan has been successfully recorded.');
    navigation.navigate('Profile'); // Navigate to desired screen after confirmation
  };

  return (
    <View style={styles.container}>
      {/* Instructional Overlay */}
      <View style={styles.instructionOverlay}>
        <Text style={styles.instructionText}>Align your head within the frame and follow the on-screen prompts.</Text>
      </View>

      {/* Contour Lines */}
      <Image
        source={require('../../assets/contourLines.png')} // Ensure this asset exists
        style={styles.contourLines}
        resizeMode="contain"
      />

      {/* Scanning Area */}
      <View style={styles.scanningArea}>
        {isScanning ? (
          <Animated.View style={[styles.pulse, { transform: [{ scale: pulseAnim }] }]} />
        ) : (
          <TouchableOpacity style={GlobalStyles(theme).buttonPrimary} onPress={startScan}>
            <Text style={styles.buttonText}>Start Scan</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Confirm Scan Button */}
      {scanValid && (
        <TouchableOpacity style={GlobalStyles(theme).buttonPrimary} onPress={confirmScan}>
          <Text style={styles.buttonText}>Confirm Scan</Text>
        </TouchableOpacity>
      )}
    </View>
  );
};

const createStyles = (theme) =>
  StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.primaryColors.inkBlack,
      justifyContent: 'center',
      alignItems: 'center',
      padding: 16,
    },
    instructionOverlay: {
      position: 'absolute',
      top: 50,
      padding: 10,
      backgroundColor: 'rgba(0, 0, 0, 0.6)',
      borderRadius: 8,
    },
    instructionText: {
      color: '#FFFFFF',
      fontSize: 16,
      fontFamily: theme.typography.secondaryFont,
      textAlign: 'center',
    },
    contourLines: {
      position: 'absolute',
      width: '80%',
      height: '80%',
    },
    scanningArea: {
      width: 200,
      height: 200,
      borderRadius: 100,
      borderWidth: 2,
      borderColor: theme.accentColors.graphiteGray,
      justifyContent: 'center',
      alignItems: 'center',
      marginBottom: 20,
    },
    pulse: {
      width: 100,
      height: 100,
      borderRadius: 50,
      backgroundColor: theme.primaryColors.deepCrimson,
      opacity: 0.7,
    },
    buttonText: {
      color: '#FFFFFF',
      fontFamily: theme.typography.secondaryFont,
      fontSize: 16,
    },
  });

export default ScanScreen;