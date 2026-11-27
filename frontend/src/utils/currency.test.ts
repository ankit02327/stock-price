import { describe, it, expect, beforeEach } from 'vitest';
import {
  convertPrice,
  formatCompactNumber,
  formatPrice,
  formatPriceDirect,
  getCurrencySymbol,
  getExchangeRate,
  setExchangeRate,
} from './currency';

describe('currency utilities', () => {
  beforeEach(() => {
    setExchangeRate(83.5);
  });

  describe('setExchangeRate and getExchangeRate', () => {
    it('updates exchange rate for valid positive finite numbers', () => {
      setExchangeRate(85.25);
      expect(getExchangeRate()).toBe(85.25);
    });

    it('ignores invalid rates such as zero, negative, NaN, and Infinity', () => {
      setExchangeRate(84.0);
      setExchangeRate(0);
      setExchangeRate(-10);
      setExchangeRate(Number.NaN);
      setExchangeRate(Number.POSITIVE_INFINITY);
      expect(getExchangeRate()).toBe(84.0);
    });
  });

  describe('convertPrice', () => {
    it('returns 0 for null, undefined, and NaN', () => {
      expect(convertPrice(null)).toBe(0);
      expect(convertPrice(undefined)).toBe(0);
      expect(convertPrice(Number.NaN)).toBe(0);
    });

    it('returns unchanged price when source and target currencies match', () => {
      expect(convertPrice(123.45, 'USD', 'USD')).toBe(123.45);
      expect(convertPrice(-50, 'INR', 'INR')).toBe(-50);
    });

    it('converts between USD and INR using explicit or configured rate', () => {
      expect(convertPrice(10, 'USD', 'INR', 80)).toBe(800);
      expect(convertPrice(800, 'INR', 'USD', 80)).toBe(10);

      setExchangeRate(85);
      expect(convertPrice(2, 'USD', 'INR')).toBe(170);
    });
  });

  describe('formatPrice', () => {
    it('formats null, undefined, and NaN as zero in the requested currency', () => {
      expect(formatPrice(null, 'USD')).toBe('$0.00');
      expect(formatPrice(undefined, 'INR')).toBe('₹0.00');
      expect(formatPrice(Number.NaN, 'USD')).toBe('$0.00');
    });

    it('formats USD and converts/formats INR with two decimals', () => {
      expect(formatPrice(1234.5, 'USD')).toBe('$1234.50');
      expect(formatPrice(1000, 'INR', 100)).toBe('₹1,00,000.00');
      expect(formatPrice(-12.5, 'USD')).toBe('$-12.50');
    });
  });

  describe('formatPriceDirect', () => {
    it('formats already-converted prices without applying exchange rate', () => {
      expect(formatPriceDirect(null, 'INR')).toBe('₹0.00');
      expect(formatPriceDirect(undefined, 'USD')).toBe('$0.00');
      expect(formatPriceDirect(100000, 'INR')).toBe('₹1,00,000.00');
      expect(formatPriceDirect(42.1, 'USD')).toBe('$42.10');
    });
  });

  describe('getCurrencySymbol and formatCompactNumber', () => {
    it('returns expected currency symbols', () => {
      expect(getCurrencySymbol('USD')).toBe('$');
      expect(getCurrencySymbol('INR')).toBe('₹');
    });

    it('formats compact numbers and handles nullish/NaN inputs', () => {
      expect(formatCompactNumber(null)).toBe('N/A');
      expect(formatCompactNumber(undefined)).toBe('N/A');
      expect(formatCompactNumber(Number.NaN)).toBe('N/A');
      expect(formatCompactNumber(1500)).toBe('1.5K');
      expect(formatCompactNumber(-2500000)).toBe('-2.5M');
    });
  });
});
