// PaymentScreen.js
import React, { useContext, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Image, ActivityIndicator, Alert, ScrollView } from 'react-native';
import { ThemeContext } from '../config/ThemeContext';
import { GlobalStyles } from '../config/GlobalStyles';
import Stripe from 'tipsi-stripe'; // Ensure Stripe is properly installed and configured
import SubscriptionBanner from '../components/SubscriptionBanner'; // Importing SubscriptionBanner

const PaymentScreen = ({ route, navigation }) => {
  const { theme } = useContext(ThemeContext);
  const styles = createStyles(theme);

  const { plan } = route.params;

  const [isProcessing, setIsProcessing] = useState(false);

  const handlePayment = async () => {
    setIsProcessing(true);
    try {
      // Initialize Stripe
      Stripe.setOptions({
        publishableKey: 'your-publishable-key', // Replace with your actual Stripe publishable key
        androidPayMode: 'test', // Change to 'production' in production environment
      });

      // Create a payment request
      const paymentMethod = await Stripe.paymentRequestWithCardForm({
        // Customize the card form as needed
      });

      // Send paymentMethod to your backend for processing
      const response = await fetch('https://api.thinkapp.com/payment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          paymentMethodId: paymentMethod.id,
          planId: plan.id,
        }),
      });

      const paymentResult = await response.json();

      if (paymentResult.success) {
        Alert.alert('Payment Success', 'Your subscription has been activated.');
        navigation.navigate('Profile');
      } else {
        Alert.alert('Payment Failed', paymentResult.message || 'Something went wrong.');
      }
    } catch (error) {
      Alert.alert('Payment Error', error.message || 'An unexpected error occurred.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      {/* Subscription Banner */}
      <SubscriptionBanner navigation={navigation} />

      {/* Secure Payment Visuals */}
      <Text style={styles.heading}>Secure Payment</Text>
      <Image
        source={require('../../assets/secure_payment.png')} // Ensure this asset exists
        style={styles.trustImage}
        resizeMode="contain"
      />
      <Text style={styles.infoText}>Your payment is processed securely.</Text>

      {/* Payment Details */}
      <View style={styles.paymentDetails}>
        <Text style={styles.planName}>{plan.name} Plan</Text>
        <Text style={styles.planPrice}>{plan.price}</Text>
        <View style={styles.benefitsContainer}>
          {plan.benefits.map((benefit, index) => (
            <Text key={index} style={styles.benefitText}>• {benefit}</Text>
          ))}
        </View>
      </View>

      {/* Payment Button */}
      <TouchableOpacity
        style={styles.subscribeButton}
        onPress={handlePayment}
        disabled={isProcessing}
      >
        {isProcessing ? (
          <ActivityIndicator size="small" color="#FFFFFF" />
        ) : (
          <Text style={styles.buttonText}>Proceed to Payment</Text>
        )}
      </TouchableOpacity>

      {/* Start Free Trial CTA */}
      <TouchableOpacity
        style={styles.freeTrialButton}
        onPress={() => {
          // Handle start free trial
          navigation.navigate('Payments', { plan: { id: 1, name: 'Free', price: '$0/month', benefits: ['Basic Tattoo Designs', 'Limited AI Features'] } });
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
      backgroundColor: theme.accentColors.warmBeige, // Warm Beige background
      padding: 16,
      alignItems: 'center',
    },
    heading: {
      fontSize: 24,
      fontWeight: 'bold',
      color: theme.primaryColors.deepCrimson,
      fontFamily: theme.typography.primaryFont,
      textAlign: 'center',
      marginBottom: 20,
    },
    trustImage: {
      width: 100,
      height: 100,
      marginBottom: 20,
    },
    infoText: {
      fontSize: 16,
      color: theme.primaryColors.inkBlack,
      fontFamily: theme.typography.secondaryFont,
      textAlign: 'center',
      marginBottom: 20,
    },
    paymentDetails: {
      width: '100%',
      backgroundColor: '#FFFFFF',
      borderRadius: 10,
      padding: 20,
      marginBottom: 20,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 5,
      elevation: 3,
    },
    planName: {
      fontSize: 20,
      fontWeight: 'bold',
      color: theme.primaryColors.inkBlack,
      fontFamily: theme.typography.primaryFont,
      marginBottom: 5,
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
      marginTop: 10,
    },
    benefitText: {
      fontSize: 14,
      color: theme.primaryColors.inkBlack,
      fontFamily: theme.typography.secondaryFont,
      marginBottom: 5,
    },
    subscribeButton: {
      backgroundColor: theme.primaryColors.deepCrimson,
      padding: 15,
      borderRadius: 8,
      alignItems: 'center',
      justifyContent: 'center',
      width: '100%',
      marginBottom: 10,
    },
    freeTrialButton: {
      backgroundColor: theme.primaryColors.deepCrimson,
      padding: 15,
      borderRadius: 8,
      alignItems: 'center',
      justifyContent: 'center',
      width: '100%',
      marginTop: 10,
    },
    buttonText: {
      color: '#FFFFFF',
      fontFamily: theme.typography.secondaryFont,
      fontSize: 16,
      fontWeight: 'bold',
    },
  });

export default PaymentScreen;
