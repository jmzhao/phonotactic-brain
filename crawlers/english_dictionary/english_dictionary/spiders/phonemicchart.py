# -*- coding: utf-8 -*-
import scrapy


class PhonemicchartSpider(scrapy.Spider):
    name = "phonemicchart"
    allowed_domains = ["phonemicchart.com"]
    custom_settings = {
    #   'CLOSESPIDER_PAGECOUNT' : 5,
      'DOWNLOAD_DELAY' : 0.5,    # 250 ms of delay
    }

    def __init__(self, wordlist=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = ['http://www.phonemicchart.com/transcribe/%s.html' % wordlist]

    def parse(self, response):
        for word_query in response.css('div.container div.main a::attr(href)').extract() :
            yield scrapy.Request(
                response.urljoin(word_query),
                callback=self.parse_word)

    def parse_word(self, response) :
        yield {
          'word' : response.css('div.main b::text').extract()[-1],
          'phonemic-script' : response.css('div.main center span::text').extract_first(),
        }
