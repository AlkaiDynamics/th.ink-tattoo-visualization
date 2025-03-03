import React, { useEffect, useState } from 'react';
import { View, Text, FlatList, StyleSheet, ActivityIndicator, Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

const PaymentsScreen = () => {
  const [payments, setPayments] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchPayments = async () => {
    try {
      const token = await AsyncStorage.getItem('token');
      const response = await fetch('http://localhost:8000/payments/', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      });

      const data = await response.json();

      if (response.ok) {
        setPayments(data);
      } else {
        Alert.alert('Error', data.detail || 'Failed to fetch payments.');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to connect to the server.');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPayments();
  }, []);

  const renderPayment = ({ item }) => (
    <View style={styles.paymentContainer}>
      <Text style={styles.amount}>${item.amount.toFixed(2)} {item.currency}</Text>
      <Text>Status: {item.status}</Text>
      <Text>Created At: {new Date(item.created_at).toLocaleString()}</Text>
      {item.updated_at && <Text>Updated At: {new Date(item.updated_at).toLocaleString()}</Text>}
    </View>
  );

  if (loading) {
    return (
      <View style={styles.loaderContainer}>
        <ActivityIndicator size="large" color="#0000ff" />
        <Text>Loading Payments...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={payments}
        keyExtractor={(item) => item.id.toString()}
        renderItem={renderPayment}
        contentContainerStyle={styles.list}
        ListEmptyComponent={<Text>No payments found.</Text>}
      />
    </View>
  );
};

export default PaymentsScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
  paymentContainer: {
    marginBottom: 16,
    padding: 16,
    borderRadius: 8,
    backgroundColor: '#f2f2f2',
    elevation: 1,
  },
  amount: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  loaderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  list: {
    paddingBottom: 16,
  },
});