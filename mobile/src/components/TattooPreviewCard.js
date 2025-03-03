// TattooPreviewCard.js
import React, { useContext, useState } from 'react';
import { View, Text, Image, StyleSheet, TouchableOpacity, Modal } from 'react-native';
import { ThemeContext } from '../config/ThemeContext';
import { GlobalStyles } from '../config/GlobalStyles';

const TattooPreviewCard = ({ tattoo, onApply }) => {
  const { theme } = useContext(ThemeContext);
  const styles = createStyles(theme);
  const [modalVisible, setModalVisible] = useState(false);

  const handleLongPress = () => {
    setModalVisible(true);
  };

  const closeModal = () => {
    setModalVisible(false);
  };

  return (
    <TouchableOpacity
      style={styles.cardContainer}
      onPress={onApply}
      onLongPress={handleLongPress}
    >
      <Image source={{ uri: tattoo.imageUrl }} style={styles.tattooImage} />
      
      {/* Modal for Artist Details */}
      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={closeModal}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Artist Details</Text>
            {/* Replace the following with actual artist details */}
            <Text style={styles.modalText}>Name: Jane Smith</Text>
            <Text style={styles.modalText}>Specialty: Watercolor Tattoos</Text>
            <Text style={styles.modalText}>Experience: 5 Years</Text>
            <TouchableOpacity
              style={GlobalStyles(theme).buttonPrimary}
              onPress={closeModal}
            >
              <Text style={styles.buttonText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </TouchableOpacity>
  );
};

const createStyles = (theme) =>
  StyleSheet.create({
    cardContainer: {
      backgroundColor: '#FFFFFF',
      borderRadius: 10,
      padding: 10,
      margin: 5,
      alignItems: 'center',
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 5,
      elevation: 3,
      width: 150,
    },
    tattooImage: {
      width: '100%',
      height: 100,
      borderRadius: 8,
      marginBottom: 10,
    },
    modalOverlay: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
    },
    modalContent: {
      width: '80%',
      backgroundColor: '#FFFFFF',
      borderRadius: 10,
      padding: 20,
      alignItems: 'center',
    },
    modalTitle: {
      fontSize: 18,
      fontWeight: 'bold',
      marginBottom: 10,
      color: theme.primaryColors.deepCrimson,
      fontFamily: theme.typography.primaryFont,
    },
    modalText: {
      fontSize: 16,
      marginBottom: 5,
      color: theme.primaryColors.inkBlack,
      fontFamily: theme.typography.secondaryFont,
    },
    buttonText: {
      color: '#FFFFFF',
      fontFamily: theme.typography.secondaryFont,
      fontSize: 16,
      fontWeight: 'bold',
    },
  });

export default TattooPreviewCard;