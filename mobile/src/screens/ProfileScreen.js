// ProfileScreen.js
import React, { useContext, useState } from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity, FlatList, Switch, Alert, ScrollView } from 'react-native';
import { ThemeContext } from '../config/ThemeContext';
import { GlobalStyles } from '../config/GlobalStyles';
import SubscriptionBanner from '../components/SubscriptionBanner'; // Importing SubscriptionBanner

const ProfileScreen = ({ navigation }) => {
  const { theme, toggleTheme } = useContext(ThemeContext);
  const styles = createStyles(theme);

  const [isDarkMode, setIsDarkMode] = useState(theme.mode === 'dark');
  const [language, setLanguage] = useState('English');
  const [isPrivacyEnabled, setIsPrivacyEnabled] = useState(true);

  const user = {
    name: 'John Doe',
    avatar: 'https://example.com/avatar.jpg', // Replace with actual user avatar URL
    savedTattoos: [
      // Array of tattoo objects
      { id: 1, imageUrl: 'https://example.com/tattoo1.jpg' },
      { id: 2, imageUrl: 'https://example.com/tattoo2.jpg' },
      { id: 3, imageUrl: 'https://example.com/tattoo3.jpg' },
      { id: 4, imageUrl: 'https://example.com/tattoo4.jpg' },
      { id: 5, imageUrl: 'https://example.com/tattoo5.jpg' },
      { id: 6, imageUrl: 'https://example.com/tattoo6.jpg' },
      // Add more tattoos as needed
    ],
  };

  const toggleDarkMode = () => {
    setIsDarkMode((previousState) => !previousState);
    toggleTheme();
  };

  const togglePrivacy = () => {
    setIsPrivacyEnabled((previousState) => !previousState);
    // Handle privacy settings logic here
    Alert.alert('Privacy Settings', `Privacy has been ${isPrivacyEnabled ? 'disabled' : 'enabled'}.`);
  };

  const renderTattoo = ({ item }) => (
    <Image source={{ uri: item.imageUrl }} style={styles.tattooImage} />
  );

  return (
    <ScrollView style={styles.container}>
      {/* Subscription Banner */}
      <SubscriptionBanner navigation={navigation} />

      {/* Profile Overview */}
      <View style={styles.profileOverview}>
        <Image source={{ uri: user.avatar }} style={styles.avatar} />
        <Text style={styles.username}>{user.name}</Text>
      </View>

      {/* Settings Toggles */}
      <View style={styles.settingsContainer}>
        <Text style={styles.sectionTitle}>Settings</Text>
        <View style={styles.settingItem}>
          <Text style={styles.settingText}>Dark Mode</Text>
          <Switch
            value={isDarkMode}
            onValueChange={toggleDarkMode}
            trackColor={{ false: theme.accentColors.graphiteGray, true: theme.primaryColors.electricBlue }}
            thumbColor="#FFFFFF"
          />
        </View>
        <View style={styles.settingItem}>
          <Text style={styles.settingText}>Language</Text>
          <TouchableOpacity onPress={() => Alert.alert('Language Selection', 'Language settings not implemented yet.')}>
            <Text style={styles.settingDetail}>{language}</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.settingItem}>
          <Text style={styles.settingText}>Privacy</Text>
          <Switch
            value={isPrivacyEnabled}
            onValueChange={togglePrivacy}
            trackColor={{ false: theme.accentColors.graphiteGray, true: theme.primaryColors.electricBlue }}
            thumbColor="#FFFFFF"
          />
        </View>
      </View>

      {/* Saved Tattoos */}
      <View style={styles.savedTattoosContainer}>
        <Text style={styles.sectionTitle}>Saved Tattoos</Text>
        {user.savedTattoos.length > 0 ? (
          <FlatList
            data={user.savedTattoos}
            renderItem={renderTattoo}
            keyExtractor={(item) => item.id.toString()}
            numColumns={3}
            contentContainerStyle={styles.tattoosList}
          />
        ) : (
          <Text style={styles.noTattoosText}>No saved tattoos yet.</Text>
        )}
      </View>
    </ScrollView>
  );
};

const createStyles = (theme) =>
  StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.primaryColors.inkBlack,
      padding: 16,
    },
    profileOverview: {
      alignItems: 'center',
      marginBottom: 20,
    },
    avatar: {
      width: 100,
      height: 100,
      borderRadius: 50,
      marginBottom: 10,
    },
    username: {
      fontSize: 20,
      fontWeight: 'bold',
      color: theme.primaryColors.deepCrimson,
      fontFamily: theme.typography.primaryFont,
    },
    settingsContainer: {
      marginBottom: 20,
      backgroundColor: theme.accentColors.graphiteGray,
      borderRadius: 10,
      padding: 15,
    },
    sectionTitle: {
      fontSize: 18,
      fontWeight: 'bold',
      color: theme.primaryColors.deepCrimson,
      fontFamily: theme.typography.primaryFont,
      marginBottom: 10,
    },
    settingItem: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: 15,
    },
    settingText: {
      fontSize: 16,
      color: theme.primaryColors.inkBlack,
      fontFamily: theme.typography.secondaryFont,
    },
    settingDetail: {
      fontSize: 16,
      color: theme.primaryColors.electricBlue,
      fontFamily: theme.typography.secondaryFont,
    },
    savedTattoosContainer: {
      flex: 1,
    },
    tattoosList: {
      alignItems: 'center',
    },
    tattooImage: {
      width: 100,
      height: 100,
      borderRadius: 8,
      margin: 5,
    },
    noTattoosText: {
      fontSize: 16,
      color: theme.primaryColors.inkBlack,
      fontFamily: theme.typography.secondaryFont,
      textAlign: 'center',
      marginTop: 20,
    },
  });

export default ProfileScreen;