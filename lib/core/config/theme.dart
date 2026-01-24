import 'package:flutter/material.dart';

/// G20 Platform color palette
class G20Colors {
  // Primary colors
  static const Color primary = Color(0xFF1E88E5);
  static const Color primaryDark = Color(0xFF1565C0);
  static const Color primaryLight = Color(0xFF42A5F5);

  // Background colors for dark theme (RF visualization friendly)
  static const Color backgroundDark = Color(0xFF0D1117);
  static const Color surfaceDark = Color(0xFF161B22);
  static const Color cardDark = Color(0xFF21262D);

  // Background colors for light theme
  static const Color backgroundLight = Color(0xFFF6F8FA);
  static const Color surfaceLight = Color(0xFFFFFFFF);
  static const Color cardLight = Color(0xFFF3F4F6);

  // Detection colors (class-based)
  static const Color detectionGreen = Color(0xFF4CAF50);
  static const Color detectionRed = Color(0xFFF44336);
  static const Color detectionYellow = Color(0xFFFFEB3B);
  static const Color detectionOrange = Color(0xFFFF9800);
  static const Color detectionPurple = Color(0xFF9C27B0);
  static const Color detectionCyan = Color(0xFF00BCD4);

  // Status colors
  static const Color success = Color(0xFF4CAF50);
  static const Color warning = Color(0xFFFF9800);
  static const Color error = Color(0xFFF44336);
  static const Color info = Color(0xFF2196F3);

  // Text colors
  static const Color textPrimaryDark = Color(0xFFE6EDF3);
  static const Color textSecondaryDark = Color(0xFF8B949E);
  static const Color textPrimaryLight = Color(0xFF24292F);
  static const Color textSecondaryLight = Color(0xFF57606A);

  // Waterfall colormap endpoints
  static const Color waterfallLow = Color(0xFF000033);
  static const Color waterfallMid = Color(0xFF0066CC);
  static const Color waterfallHigh = Color(0xFFFFFF00);
  static const Color waterfallMax = Color(0xFFFF0000);
}

/// G20 Platform theme definitions
class G20Theme {
  static ThemeData get dark {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      colorScheme: ColorScheme.dark(
        primary: G20Colors.primary,
        secondary: G20Colors.primaryLight,
        surface: G20Colors.surfaceDark,
        error: G20Colors.error,
      ),
      scaffoldBackgroundColor: G20Colors.backgroundDark,
      cardColor: G20Colors.cardDark,
      appBarTheme: const AppBarTheme(
        backgroundColor: G20Colors.surfaceDark,
        foregroundColor: G20Colors.textPrimaryDark,
        elevation: 0,
      ),
      cardTheme: CardThemeData(
        color: G20Colors.cardDark,
        elevation: 2,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: G20Colors.cardDark,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: G20Colors.textSecondaryDark, width: 0.5),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: G20Colors.primary, width: 2),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: G20Colors.primary,
          foregroundColor: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
        ),
      ),
      textTheme: const TextTheme(
        headlineLarge: TextStyle(
          color: G20Colors.textPrimaryDark,
          fontWeight: FontWeight.bold,
        ),
        headlineMedium: TextStyle(
          color: G20Colors.textPrimaryDark,
          fontWeight: FontWeight.bold,
        ),
        titleLarge: TextStyle(
          color: G20Colors.textPrimaryDark,
          fontWeight: FontWeight.w600,
        ),
        titleMedium: TextStyle(
          color: G20Colors.textPrimaryDark,
        ),
        bodyLarge: TextStyle(
          color: G20Colors.textPrimaryDark,
        ),
        bodyMedium: TextStyle(
          color: G20Colors.textSecondaryDark,
        ),
        labelLarge: TextStyle(
          color: G20Colors.textPrimaryDark,
          fontWeight: FontWeight.w500,
        ),
      ),
      dividerTheme: const DividerThemeData(
        color: G20Colors.textSecondaryDark,
        thickness: 0.5,
      ),
      navigationRailTheme: NavigationRailThemeData(
        backgroundColor: G20Colors.surfaceDark,
        selectedIconTheme: const IconThemeData(color: G20Colors.primary),
        unselectedIconTheme: const IconThemeData(color: G20Colors.textSecondaryDark),
        selectedLabelTextStyle: const TextStyle(color: G20Colors.primary),
        unselectedLabelTextStyle: const TextStyle(color: G20Colors.textSecondaryDark),
      ),
    );
  }

  static ThemeData get light {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.light,
      colorScheme: ColorScheme.light(
        primary: G20Colors.primary,
        secondary: G20Colors.primaryDark,
        surface: G20Colors.surfaceLight,
        error: G20Colors.error,
      ),
      scaffoldBackgroundColor: G20Colors.backgroundLight,
      cardColor: G20Colors.cardLight,
      appBarTheme: const AppBarTheme(
        backgroundColor: G20Colors.surfaceLight,
        foregroundColor: G20Colors.textPrimaryLight,
        elevation: 0,
      ),
      cardTheme: CardThemeData(
        color: G20Colors.cardLight,
        elevation: 2,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: G20Colors.surfaceLight,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide.none,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: G20Colors.textSecondaryLight, width: 0.5),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: const BorderSide(color: G20Colors.primary, width: 2),
        ),
      ),
      textTheme: const TextTheme(
        headlineLarge: TextStyle(
          color: G20Colors.textPrimaryLight,
          fontWeight: FontWeight.bold,
        ),
        headlineMedium: TextStyle(
          color: G20Colors.textPrimaryLight,
          fontWeight: FontWeight.bold,
        ),
        titleLarge: TextStyle(
          color: G20Colors.textPrimaryLight,
          fontWeight: FontWeight.w600,
        ),
        titleMedium: TextStyle(
          color: G20Colors.textPrimaryLight,
        ),
        bodyLarge: TextStyle(
          color: G20Colors.textPrimaryLight,
        ),
        bodyMedium: TextStyle(
          color: G20Colors.textSecondaryLight,
        ),
        labelLarge: TextStyle(
          color: G20Colors.textPrimaryLight,
          fontWeight: FontWeight.w500,
        ),
      ),
    );
  }
}
