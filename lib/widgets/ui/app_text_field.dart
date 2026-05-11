import 'package:flutter/material.dart';

/// Thin wrapper over `TextFormField` that picks up FitNova's
/// `inputDecorationTheme` (outlined, primary-focused border, surface fill).
class AppTextField extends StatelessWidget {
  final TextEditingController? controller;
  final String? label;
  final String? hint;
  final String? helper;
  final String? errorText;
  final TextInputType? keyboardType;
  final TextInputAction? textInputAction;
  final bool obscure;
  final IconData? prefixIcon;
  final Widget? suffix;
  final ValueChanged<String>? onChanged;
  final FormFieldValidator<String>? validator;
  final int? maxLines;
  final int? minLines;
  final bool autofocus;
  final bool enabled;
  final FocusNode? focusNode;
  final ValueChanged<String>? onSubmitted;

  const AppTextField({
    super.key,
    this.controller,
    this.label,
    this.hint,
    this.helper,
    this.errorText,
    this.keyboardType,
    this.textInputAction,
    this.obscure = false,
    this.prefixIcon,
    this.suffix,
    this.onChanged,
    this.validator,
    this.maxLines = 1,
    this.minLines,
    this.autofocus = false,
    this.enabled = true,
    this.focusNode,
    this.onSubmitted,
  });

  @override
  Widget build(BuildContext context) {
    return TextFormField(
      controller: controller,
      focusNode: focusNode,
      enabled: enabled,
      autofocus: autofocus,
      keyboardType: keyboardType,
      textInputAction: textInputAction,
      obscureText: obscure,
      onChanged: onChanged,
      validator: validator,
      onFieldSubmitted: onSubmitted,
      maxLines: obscure ? 1 : maxLines,
      minLines: minLines,
      decoration: InputDecoration(
        labelText: label,
        hintText: hint,
        helperText: helper,
        errorText: errorText,
        prefixIcon: prefixIcon != null ? Icon(prefixIcon) : null,
        suffix: suffix,
      ),
    );
  }
}
