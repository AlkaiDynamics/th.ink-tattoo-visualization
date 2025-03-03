// SubscriptionScreen.js
import React, { useContext } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Image, ScrollView } from 'react-native';
import { ThemeContext } from '../config/ThemeContext';
import { GlobalStyles } from '../config/GlobalStyles';

const SubscriptionScreen = ({ navigation }) => {
  const { theme } = useContext(ThemeContext);
  const styles = createStyles(theme);

  const subscriptionPlans = [
    {
      id: 1,
      name: 'Free',
      price: '$0/month',
      benefits: ['Basic Tattoo Designs', 'Limited AI Features'],
      isBestValue: false,
    },
    {
      id: 2,
      name: 'Premium',
      price: '$9.99/month',
      benefits: ['Unlimited AI Designs', 'Exclusive Content', 'Priority Support'],
      isBestValue: true,
    },
    {
      id: 3,
      name: 'Yearly',
      price: '$99.99/year',
      benefits: ['All Premium Features', '2 Free Tattoo Consultations', 'Lifetime Access'],
      isBestValue: false,
    },
  ];

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.heading}>Choose Your Subscription</Text>
      {subscriptionPlans.map((plan) => (
        <View key={plan.id} style={styles.planCard}>
          {plan.isBestValue && (
            <Image
              source={require('../../assets/best_value_badge.png')} // Ensure this asset exists
              style={styles.badge}
              resizeMode="contain"
            />
          )}
          <Text style={styles.planName}>{plan.name}</Text>
          <Text style={styles.planPrice}>{plan.price}</Text>
          <View style={styles.benefitsContainer}>
            {plan.benefits.map((benefit, index) => (
              <Text key={index} style={styles.benefitText}>• {benefit}</Text>
            ))}
          </View>
          <TouchableOpacity
            style={styles.subscribeButton}
            onPress={() => {
              // Navigate to payment screen with selected plan
              navigation.navigate('Payment', { plan });
            }}
          >
            <Text style={styles.buttonText}>Subscribe</Text>
          </TouchableOpacity>
        </View>
      ))}
      <TouchableOpacity
        style={styles.freeTrialButton}
        onPress={() => {
          // Handle start free trial
          navigation.navigate('Payments', { plan: subscriptionPlans[0] });
        }}
      >
        <Text style={styles.buttonText}>Start Free Trial</Text>
      </TouchableOpacity>
    </ScrollView>
  );
};

const createStyles = (theme) =>
  StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.accentColors.warmBeige,
      padding: 16,
    },
    heading: {
      fontSize: 24,
      fontWeight: 'bold',
      color: theme.primaryColors.deepCrimson,
      fontFamily: theme.typography.primaryFont,
      textAlign: 'center',
      marginBottom: 20,
    },
    planCard: {
      backgroundColor: '#FFFFFF',
      borderRadius: 10,
      padding: 20,
      marginBottom: 20,
      position: 'relative',
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 5,
      elevation: 3,
    },
    badge: {
      position: 'absolute',
      top: -10,
      right: -10,
      width: 50,
      height: 50,
    },
    planName: {
      fontSize: 20,
      fontWeight: 'bold',
      color: theme.primaryColors.inkBlack,
      fontFamily: theme.typography.primaryFont,
      marginBottom: 10,
      textAlign: 'center',
    },
    planPrice: {
      fontSize: 18,
      color: theme.primaryColors.deepCrimson,
      fontFamily: theme.typography.primaryFont,
      marginBottom: 10,
      textAlign: 'center',
    },
    benefitsContainer: {
      marginBottom: 15,
    },
    benefitText: {
      fontSize: 14,
      color: theme.primaryColors.inkBlack,
      fontFamily: theme.typography.secondaryFont,
      marginBottom: 5,
    },
    subscribeButton: {
      backgroundColor: theme.primaryColors.deepCrimson,
      padding: 12,
      borderRadius: 8,
      alignItems: 'center',
      justifyContent: 'center',
    },
    freeTrialButton: {
      backgroundColor: theme.primaryColors.deepCrimson,
      padding: 15,
      borderRadius: 8,
      alignItems: 'center',
      justifyContent: 'center',
      marginTop: 10,
    },
    buttonText: {
      color: '#FFFFFF',
      fontFamily: theme.typography.secondaryFont,
      fontSize: 16,
      fontWeight: 'bold',
    },
  });

export default SubscriptionScreen;