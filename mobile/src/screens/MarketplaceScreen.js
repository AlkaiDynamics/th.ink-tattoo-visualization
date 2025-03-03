// MarketplaceScreen.js
import React, { useContext, useState, useEffect } from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity, Image, Picker } from 'react-native';
import { ThemeContext } from '../config/ThemeContext';
import { GlobalStyles } from '../config/GlobalStyles';
import TattooPreviewCard from '../components/TattooPreviewCard';
import axios from 'axios';

const categories = ['Featured', 'Traditional', 'Minimalist', 'Watercolor'];

const MarketplaceScreen = ({ navigation }) => {
  const { theme } = useContext(ThemeContext);
  const styles = createStyles(theme);

  const [selectedCategory, setSelectedCategory] = useState('Featured');
  const [sortOption, setSortOption] = useState('Popularity');
  const [tattoos, setTattoos] = useState([]);
  const [artists, setArtists] = useState([]);

  useEffect(() => {
    fetchTattoos();
    fetchArtists();
  }, [selectedCategory, sortOption]);

  const fetchTattoos = async () => {
    try {
      const response = await axios.get('https://api.thinkapp.com/marketplace/tattoos', {
        params: {
          category: selectedCategory,
          sort: sortOption.toLowerCase(),
        },
      });
      setTattoos(response.data.tattoos);
    } catch (error) {
      console.error('Error fetching tattoos:', error);
    }
  };

  const fetchArtists = async () => {
    try {
      const response = await axios.get('https://api.thinkapp.com/marketplace/artists');
      setArtists(response.data.artists);
    } catch (error) {
      console.error('Error fetching artists:', error);
    }
  };

  const renderTattoo = ({ item }) => (
    <TattooPreviewCard
      tattoo={item}
      onApply={() => navigation.navigate('ARViewer', { tattoo: item.imageUrl })}
    />
  );

  const renderArtist = ({ item }) => (
    <View style={styles.artistCard}>
      <Image source={{ uri: item.profileImage }} style={styles.artistImage} />
      <Text style={styles.artistName}>{item.name}</Text>
      <TouchableOpacity
        style={GlobalStyles(theme).buttonPrimary}
        onPress={() => {
          // Navigate to booking screen or initiate booking process
          Alert.alert('Booking', `Booking with ${item.name} is not implemented yet.`);
        }}
      >
        <Text style={styles.buttonText}>Book Now</Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <View style={styles.container}>
      {/* Category Tabs */}
      <View style={styles.tabsContainer}>
        {categories.map((category) => (
          <TouchableOpacity
            key={category}
            style={[
              styles.tab,
              selectedCategory === category && styles.activeTab,
            ]}
            onPress={() => setSelectedCategory(category)}
          >
            <Text
              style={[
                styles.tabText,
                selectedCategory === category && styles.activeTabText,
              ]}
            >
              {category}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Filter & Sort Options */}
      <View style={styles.filterSortContainer}>
        <Picker
          selectedValue={sortOption}
          style={styles.picker}
          onValueChange={(itemValue) => setSortOption(itemValue)}
        >
          <Picker.Item label="Popularity" value="Popularity" />
          <Picker.Item label="Price: Low to High" value="Price: Low to High" />
          <Picker.Item label="Price: High to Low" value="Price: High to Low" />
        </Picker>
      </View>

      {/* Tattoo Listings */}
      <FlatList
        data={tattoos}
        renderItem={renderTattoo}
        keyExtractor={(item) => item.id.toString()}
        contentContainerStyle={styles.tattoosList}
      />

      {/* Artist Spotlight */}
      <Text style={styles.sectionTitle}>Featured Artists</Text>
      <FlatList
        data={artists}
        renderItem={renderArtist}
        keyExtractor={(item) => item.id.toString()}
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.artistsList}
      />
    </View>
  );
};

const createStyles = (theme) =>
  StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.primaryColors.inkBlack,
      padding: 16,
    },
    tabsContainer: {
      flexDirection: 'row',
      justifyContent: 'space-around',
      marginBottom: 10,
    },
    tab: {
      paddingVertical: 8,
      paddingHorizontal: 16,
      borderRadius: 20,
      backgroundColor: theme.accentColors.graphiteGray,
    },
    activeTab: {
      backgroundColor: theme.primaryColors.deepCrimson,
    },
    tabText: {
      color: '#FFFFFF',
      fontFamily: theme.typography.secondaryFont,
      fontSize: 14,
    },
    activeTabText: {
      fontWeight: 'bold',
    },
    filterSortContainer: {
      flexDirection: 'row',
      justifyContent: 'flex-end',
      marginBottom: 10,
    },
    picker: {
      height: 50,
      width: 200,
      color: '#FFFFFF',
      backgroundColor: theme.accentColors.graphiteGray,
    },
    tattoosList: {
      paddingBottom: 20,
    },
    sectionTitle: {
      fontSize: 20,
      fontWeight: 'bold',
      color: theme.primaryColors.deepCrimson,
      fontFamily: theme.typography.primaryFont,
      marginVertical: 10,
    },
    artistsList: {
      paddingVertical: 10,
    },
    artistCard: {
      width: 150,
      backgroundColor: theme.accentColors.warmBeige,
      borderRadius: 10,
      padding: 10,
      alignItems: 'center',
      marginRight: 10,
    },
    artistImage: {
      width: 80,
      height: 80,
      borderRadius: 40,
      marginBottom: 10,
    },
    artistName: {
      fontSize: 16,
      fontWeight: 'bold',
      color: theme.primaryColors.inkBlack,
      fontFamily: theme.typography.secondaryFont,
      marginBottom: 10,
      textAlign: 'center',
    },
    buttonText: {
      color: '#FFFFFF',
      fontFamily: theme.typography.secondaryFont,
      fontSize: 14,
    },
  });

export default MarketplaceScreen;