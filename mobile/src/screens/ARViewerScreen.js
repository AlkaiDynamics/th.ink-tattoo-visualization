// ARViewerScreen.js
import React, { useContext, useState } from 'react';
import { View, Text, TouchableOpacity, Slider, StyleSheet } from 'react-native';
import { ARKit } from 'react-native-arkit';
import { ThemeContext } from '../config/ThemeContext';
import { GlobalStyles } from '../config/GlobalStyles';

const ARViewerScreen = ({ navigation }) => {
  const { theme } = useContext(ThemeContext);
  const styles = createStyles(theme);
  const [opacity, setOpacity] = useState(1);

  const handleApplyAnother = () => {
    // Logic to apply another tattoo
  };

  const handleFinalize = () => {
    // Logic to finalize and save the tattoo
  };

  return (
    <View style={styles.container}>
      <ARKit
        style={styles.arView}
        planeDetection={ARKit.ARPlaneDetection.Horizontal}
        lightEstimationEnabled
        onPlaneDetected={(anchor) => {
          // Handle plane detection
        }}
      >
        {/* Tattoo rendering with dynamic shadowing & occlusion */}
        <ARKit.Model
          model={{
            scale: [0.1, 0.1, 0.1],
            position: [0, 0, -0.5],
            file: 'tattoo.usdz', // Ensure the tattoo model is available in assets
          }}
          materials={{
            billboard: {
              opacity: opacity,
            },
          }}
          onPinchGesture={() => {
            // Handle pinch to resize
          }}
          onDrag={(x, y, z) => {
            // Handle drag to move
          }}
          onRotate={(angle) => {
            // Handle rotate gesture
          }}
        />
      </ARKit>
      <View style={styles.sliderContainer}>
        <Text style={styles.sliderLabel}>Tattoo Opacity:</Text>
        <Slider
          style={styles.slider}
          minimumValue={0}
          maximumValue={1}
          value={opacity}
          onValueChange={(value) => setOpacity(value)}
          minimumTrackTintColor={theme.primaryColors.electricBlue}
          maximumTrackTintColor={theme.accentColors.graphiteGray}
        />
      </View>
      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={GlobalStyles(theme).buttonPrimary}
          onPress={handleApplyAnother}
        >
          <Text style={styles.buttonText}>Apply Another Tattoo</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={GlobalStyles(theme).buttonPrimary}
          onPress={handleFinalize}
        >
          <Text style={styles.buttonText}>Finalize & Save</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const createStyles = (theme) =>
  StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.primaryColors.inkBlack,
      justifyContent: 'flex-end',
      alignItems: 'center',
    },
    arView: {
      flex: 1,
      width: '100%',
    },
    sliderContainer: {
      width: '90%',
      padding: 10,
      backgroundColor: theme.accentColors.warmBeige,
      borderRadius: 8,
      marginBottom: 10,
    },
    sliderLabel: {
      fontSize: 16,
      fontFamily: theme.typography.secondaryFont,
      color: theme.primaryColors.inkBlack,
      marginBottom: 5,
    },
    slider: {
      width: '100%',
      height: 40,
    },
    buttonContainer: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      width: '90%',
      marginBottom: 20,
    },
    buttonText: {
      color: '#FFFFFF',
      fontFamily: theme.typography.secondaryFont,
      fontSize: 16,
    },
  });

export default ARViewerScreen;