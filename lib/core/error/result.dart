/// Result Type - Railway-oriented error handling for G20
/// 
/// Features:
/// - Type-safe error handling without exceptions
/// - Chainable transformations (map, flatMap)
/// - Pattern matching for success/failure
/// - Async support
/// 
/// Usage:
///   Result<User, ApiError> result = await api.getUser(id);
///   result.fold(
///     onSuccess: (user) => showUser(user),
///     onFailure: (error) => showError(error),
///   );
library;

import 'dart:async';

/// Sealed Result type - either Success<T> or Failure<E>
sealed class Result<T, E> {
  const Result._();

  /// Create a success result
  factory Result.success(T value) = Success<T, E>;

  /// Create a failure result
  factory Result.failure(E error) = Failure<T, E>;

  /// Create from nullable value (null becomes failure)
  factory Result.fromNullable(T? value, E Function() onNull) {
    return value != null ? Success(value) : Failure(onNull());
  }

  /// Create from try-catch
  static Result<T, E> tryCatch<T, E>(T Function() fn, E Function(Object error, StackTrace stack) onError) {
    try {
      return Success(fn());
    } catch (e, st) {
      return Failure(onError(e, st));
    }
  }

  /// Async try-catch
  static Future<Result<T, E>> tryCatchAsync<T, E>(
    Future<T> Function() fn,
    E Function(Object error, StackTrace stack) onError,
  ) async {
    try {
      return Success(await fn());
    } catch (e, st) {
      return Failure(onError(e, st));
    }
  }

  // Pattern matching
  bool get isSuccess;
  bool get isFailure;
  T? get valueOrNull;
  E? get errorOrNull;

  /// Pattern match on result
  R fold<R>({
    required R Function(T value) onSuccess,
    required R Function(E error) onFailure,
  });

  /// Transform success value
  Result<R, E> map<R>(R Function(T value) fn);

  /// Transform with result-returning function
  Result<R, E> flatMap<R>(Result<R, E> Function(T value) fn);

  /// Transform error
  Result<T, F> mapError<F>(F Function(E error) fn);

  /// Get value or default
  T getOrElse(T Function() defaultValue);

  /// Get value or throw
  T getOrThrow();

  /// Execute side effect on success
  Result<T, E> onSuccess(void Function(T value) fn);

  /// Execute side effect on failure
  Result<T, E> onFailure(void Function(E error) fn);
}

/// Success case
final class Success<T, E> extends Result<T, E> {
  final T value;
  const Success(this.value) : super._();

  @override
  bool get isSuccess => true;

  @override
  bool get isFailure => false;

  @override
  T? get valueOrNull => value;

  @override
  E? get errorOrNull => null;

  @override
  R fold<R>({
    required R Function(T value) onSuccess,
    required R Function(E error) onFailure,
  }) => onSuccess(value);

  @override
  Result<R, E> map<R>(R Function(T value) fn) => Success(fn(value));

  @override
  Result<R, E> flatMap<R>(Result<R, E> Function(T value) fn) => fn(value);

  @override
  Result<T, F> mapError<F>(F Function(E error) fn) => Success(value);

  @override
  T getOrElse(T Function() defaultValue) => value;

  @override
  T getOrThrow() => value;

  @override
  Result<T, E> onSuccess(void Function(T value) fn) {
    fn(value);
    return this;
  }

  @override
  Result<T, E> onFailure(void Function(E error) fn) => this;

  @override
  String toString() => 'Success($value)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Success<T, E> && other.value == value;

  @override
  int get hashCode => value.hashCode;
}

/// Failure case
final class Failure<T, E> extends Result<T, E> {
  final E error;
  const Failure(this.error) : super._();

  @override
  bool get isSuccess => false;

  @override
  bool get isFailure => true;

  @override
  T? get valueOrNull => null;

  @override
  E? get errorOrNull => error;

  @override
  R fold<R>({
    required R Function(T value) onSuccess,
    required R Function(E error) onFailure,
  }) => onFailure(error);

  @override
  Result<R, E> map<R>(R Function(T value) fn) => Failure(error);

  @override
  Result<R, E> flatMap<R>(Result<R, E> Function(T value) fn) => Failure(error);

  @override
  Result<T, F> mapError<F>(F Function(E error) fn) => Failure(fn(error));

  @override
  T getOrElse(T Function() defaultValue) => defaultValue();

  @override
  T getOrThrow() => throw error is Exception ? error as Exception : Exception(error.toString());

  @override
  Result<T, E> onSuccess(void Function(T value) fn) => this;

  @override
  Result<T, E> onFailure(void Function(E error) fn) {
    fn(error);
    return this;
  }

  @override
  String toString() => 'Failure($error)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Failure<T, E> && other.error == error;

  @override
  int get hashCode => error.hashCode;
}

// ============================================
// ASYNC EXTENSIONS
// ============================================

extension FutureResultExtensions<T, E> on Future<Result<T, E>> {
  /// Map success value asynchronously
  Future<Result<R, E>> mapAsync<R>(FutureOr<R> Function(T value) fn) async {
    final result = await this;
    return switch (result) {
      Success(:final value) => Success(await fn(value)),
      Failure(:final error) => Failure(error),
    };
  }

  /// FlatMap asynchronously
  Future<Result<R, E>> flatMapAsync<R>(FutureOr<Result<R, E>> Function(T value) fn) async {
    final result = await this;
    return switch (result) {
      Success(:final value) => await fn(value),
      Failure(:final error) => Failure(error),
    };
  }
}

// ============================================
// COMMON ERROR TYPES
// ============================================

/// Base class for G20 application errors
sealed class G20Error {
  final String message;
  final String? code;
  final Object? cause;
  final StackTrace? stackTrace;

  const G20Error(this.message, {this.code, this.cause, this.stackTrace});

  @override
  String toString() => code != null ? '[$code] $message' : message;
}

/// Network/API errors
final class NetworkError extends G20Error {
  final int? statusCode;
  
  const NetworkError(super.message, {super.code, super.cause, super.stackTrace, this.statusCode});

  factory NetworkError.timeout([String? message]) => 
      NetworkError(message ?? 'Request timed out', code: 'TIMEOUT');
  
  factory NetworkError.noConnection([String? message]) => 
      NetworkError(message ?? 'No network connection', code: 'NO_CONNECTION');
  
  factory NetworkError.serverError(int statusCode, [String? message]) => 
      NetworkError(message ?? 'Server error', code: 'SERVER_ERROR', statusCode: statusCode);
}

/// Hardware/SDR errors
final class HardwareError extends G20Error {
  const HardwareError(super.message, {super.code, super.cause, super.stackTrace});

  factory HardwareError.notConnected([String? message]) =>
      HardwareError(message ?? 'Hardware not connected', code: 'NOT_CONNECTED');

  factory HardwareError.tuningFailed(double freqMHz, [String? message]) =>
      HardwareError(message ?? 'Failed to tune to $freqMHz MHz', code: 'TUNE_FAILED');
}

/// Validation errors
final class ValidationError extends G20Error {
  final String? field;
  
  const ValidationError(super.message, {this.field, super.code, super.cause, super.stackTrace});

  factory ValidationError.required(String field) =>
      ValidationError('$field is required', field: field, code: 'REQUIRED');

  factory ValidationError.invalid(String field, String reason) =>
      ValidationError('$field is invalid: $reason', field: field, code: 'INVALID');
}

/// Inference/ML errors
final class InferenceError extends G20Error {
  const InferenceError(super.message, {super.code, super.cause, super.stackTrace});

  factory InferenceError.modelNotFound(String modelName) =>
      InferenceError('Model not found: $modelName', code: 'MODEL_NOT_FOUND');

  factory InferenceError.inferFailed([String? message]) =>
      InferenceError(message ?? 'Inference failed', code: 'INFER_FAILED');
}

// ============================================
// RESULT TYPE ALIASES
// ============================================

/// Commonly used result types
typedef NetworkResult<T> = Result<T, NetworkError>;
typedef HardwareResult<T> = Result<T, HardwareError>;
typedef ValidationResult<T> = Result<T, ValidationError>;
typedef InferenceResult<T> = Result<T, InferenceError>;
typedef G20Result<T> = Result<T, G20Error>;
