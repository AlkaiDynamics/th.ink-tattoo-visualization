// SubscriptionBanner.js
import React, { useContext } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Image } from 'react-native';
import { ThemeContext } from '../config/ThemeContext';
import { GlobalStyles } from '../config/GlobalStyles';

const SubscriptionBanner = ({ navigation }) => {
  const { theme } = useContext(ThemeContext);
  const styles = createStyles(theme);

  return (
    <View style={styles.bannerContainer}>
      <Image
        source={require('../../assets/premium_icon.png')} // Ensure this asset exists
        style={styles.icon}
        resizeMode="contain"
      />
      <View style={styles.textContainer}>
        <Text style={styles.bannerText}>Unlock Exclusive Features with Premium!</Text>
      </View>
      <TouchableOpacity
        style={styles.ctaButton}
        onPress={() => navigation.navigate('Subscription')}
      >
        <Text style={styles.buttonText}>Upgrade Now</Text>
      </TouchableOpacity>
    </View>
  );
};

const createStyles = (theme) =>
  StyleSheet.create({
    bannerContainer: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: theme.accentColors.warmBeige,
      padding: 10,
      borderRadius: 8,
      marginVertical: 10,
    },
    icon: {
      width: 40,
      height: 40,
      marginRight: 10,
    },
    textContainer: {
      flex: 1,
    },
    bannerText: {
      fontSize: 16,
      color: theme.primaryColors.inkBlack,
      fontFamily: theme.typography.secondaryFont,
    },
    ctaButton: {
      backgroundColor: theme.primaryColors.deepCrimson,
      paddingVertical: 8,
      paddingHorizontal: 12,
      borderRadius: 5,
    },
    buttonText: {
      color: '#FFFFFF',
      fontFamily: theme.typography.secondaryFont,
      fontSize: 14,
      fontWeight: 'bold',
    },
  });

export default SubscriptionBanner;