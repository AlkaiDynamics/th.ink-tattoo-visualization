// TattooGeneratorScreen.js
import React, { useContext, useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Image, ActivityIndicator, Alert } from 'react-native';
import { ThemeContext } from '../config/ThemeContext';
import { GlobalStyles } from '../config/GlobalStyles';
import Carousel from 'react-native-snap-carousel';
import axios from 'axios';

const TattooGeneratorScreen = ({ navigation }) => {
  const { theme } = useContext(ThemeContext);
  const styles = createStyles(theme);

  const [style, setStyle] = useState('');
  const [subject, setSubject] = useState('');
  const [attributes, setAttributes] = useState('');
  const [generatedTattoos, setGeneratedTattoos] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const validateInputs = () => {
    if (!style.trim()) {
      Alert.alert('Validation Error', 'Please enter a tattoo style.');
      return false;
    }
    if (!subject.trim()) {
      Alert.alert('Validation Error', 'Please enter a tattoo subject.');
      return false;
    }
    if (!attributes.trim()) {
      Alert.alert('Validation Error', 'Please enter specific attributes.');
      return false;
    }
    return true;
  };

  const generateTattoo = async () => {
    if (!validateInputs()) return;

    setIsLoading(true);
    try {
      const prompt = `Generate a detailed ${style} tattoo of a ${subject} with ${attributes}.`;
      // Placeholder for AI generation API call
      // Replace with actual API endpoint and parameters
      const response = await axios.post('https://api.thinkapp.com/generate-tattoo', { prompt });

      // Assuming the API returns an array of image URLs
      setGeneratedTattoos(response.data.tattoos);
    } catch (error) {
      Alert.alert('Generation Error', 'Failed to generate tattoos. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const renderTattooItem = ({ item, index }) => (
    <View style={styles.carouselItem}>
      <Image source={{ uri: item }} style={styles.tattooImage} />
      <TouchableOpacity
        style={GlobalStyles(theme).buttonPrimary}
        onPress={() => navigation.navigate('ARViewer', { tattoo: item })}
      >
        <Text style={styles.buttonText}>Apply in AR</Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.heading}>AI Tattoo Generator</Text>
      <TextInput
        style={styles.input}
        placeholder="Tattoo Style (e.g., Watercolor)"
        placeholderTextColor={theme.accentColors.graphiteGray}
        value={style}
        onChangeText={setStyle}
      />
      <TextInput
        style={styles.input}
        placeholder="Tattoo Subject (e.g., Dragon)"
        placeholderTextColor={theme.accentColors.graphiteGray}
        value={subject}
        onChangeText={setSubject}
      />
      <TextInput
        style={styles.input}
        placeholder="Specific Attributes (e.g., Soft, Blended Edges)"
        placeholderTextColor={theme.accentColors.graphiteGray}
        value={attributes}
        onChangeText={setAttributes}
      />
      <TouchableOpacity
        style={GlobalStyles(theme).buttonPrimary}
        onPress={generateTattoo}
      >
        <Text style={styles.buttonText}>Generate</Text>
      </TouchableOpacity>
      {isLoading && <ActivityIndicator size="large" color={theme.primaryColors.electricBlue} style={styles.loader} />}
      {generatedTattoos.length > 0 && (
        <Carousel
          data={generatedTattoos}
          renderItem={renderTattooItem}
          sliderWidth={300}
          itemWidth={250}
          layout={'default'}
        />
      )}
    </View>
  );
};

const createStyles = (theme) =>
  StyleSheet.create({
    container: {
      flex: 1,
      padding: 16,
      backgroundColor: theme.primaryColors.inkBlack,
      alignItems: 'center',
    },
    heading: {
      fontSize: 24,
      fontWeight: 'bold',
      marginBottom: 20,
      color: theme.primaryColors.deepCrimson,
      fontFamily: theme.typography.primaryFont,
      textAlign: 'center',
    },
    input: {
      width: '100%',
      height: 50,
      borderColor: theme.accentColors.graphiteGray,
      borderWidth: 1,
      borderRadius: 8,
      paddingHorizontal: 10,
      marginBottom: 15,
      color: '#FFFFFF',
      fontFamily: theme.typography.secondaryFont,
    },
    buttonText: {
      color: '#FFFFFF',
      fontFamily: theme.typography.secondaryFont,
      fontSize: 16,
    },
    loader: {
      marginTop: 20,
    },
    carouselItem: {
      backgroundColor: theme.accentColors.warmBeige,
      borderRadius: 8,
      padding: 10,
      alignItems: 'center',
    },
    tattooImage: {
      width: 200,
      height: 200,
      borderRadius: 8,
      marginBottom: 10,
    },
  });

export default TattooGeneratorScreen;